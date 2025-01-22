import multiprocessing as mp
import logging
import shutil
from abc import ABC, abstractmethod

import cv2

import argparse
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from external.nms import nms
from external.sort import Sort
import json
import numpy as np
import numpy.typing as npt
import os
from utils import mask_frame, parse_outputs, regionize_image
from xml.etree import ElementTree
from matplotlib.path import Path
import time
import pathlib
import torch

from strong_sort import StrongSORT, FeatureExtractor
from strongsort.yolov5.utils.general import xyxy2xywh

FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[0] / 'strongsort'  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'


LIMIT = 200


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


def mask_frame(frame: npt.NDArray, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))
    def masker(frame: npt.NDArray):
        # frame[bitmap == 0] = 0
        # frame_masked = frame & bitmap
        frame_masked = frame[tl[0]:br[0], tl[1]:br[1], :]
        return frame_masked
    setattr(masker, 't', tl[0])
    setattr(masker, 'l', tl[1])
    return masker


def get_bitmap(frame: npt.NDArray, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))

    bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br


def visualize(infilename: str, outfilename: str, masker, width, height, inQueue: mp.Queue):
    cap = cv2.VideoCapture(infilename)
    writer = cv2.VideoWriter(
        outfilename,
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)
    )
    idx = 0
    while cap.isOpened() and idx < LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        idx, tracked_objects = inQueue.get()
        if idx is None:
            break
        for tracked_object in tracked_objects:
            tl = (int(tracked_object[0]) + masker.l, int(tracked_object[1]) + masker.t)
            br = (int(tracked_object[2]) + masker.l, int(tracked_object[3]) + masker.t)
            object_index = int(tracked_object[4])
            
            frame = cv2.rectangle(frame, tl, br, colors[object_index % len(colors)], 2)
        writer.write(frame)
    cap.release()


class OpNode(ABC):

    @abstractmethod
    def __call__(self, inQueue: "mp.Queue | None", outQueue: mp.Queue) -> None:
        ...


class WriteVideo(OpNode):
    def __init__(self, outfilename: str):
        self.outfilename = outfilename
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('write')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-video-write.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        shutil.rmtree(self.outfilename, ignore_errors=True)
        os.makedirs(self.outfilename, exist_ok=True)

        idx = 0
        writer = None
        while idx < LIMIT:
            _, frame_masked, frame = inQueue.get()
            width, height = frame.shape[1], frame.shape[0]
            cv2.imwrite(os.path.join(self.outfilename, f'{idx}.jpg'), frame_masked)
            # if writer is None:
            #     logger.info((width, height))
            #     writer = cv2.VideoWriter(
            #         self.outfilename,
            #         cv2.VideoWriter_fourcc(*'mp4v'), 30, (height, width)
            #     )
            # writer.write(frame_masked)
            idx += 1
        # writer.release()



class Write(OpNode):
    def __init__(self):
        pass
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('write')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-write.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        idx = 0
        while idx < LIMIT:
            idx, tracked_objects = inQueue.get()
            if idx is None:
                break
            # logger.info(str(tracked_objects))
            logger.info(str(idx))
        # logger.info('done')


class Track(OpNode):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('track')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-track.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        with torch.no_grad():
            start = time.time()
            tracker = StrongSORT(self.width, self.height)
            frame_idx = 0
            cache = {}
            logger.info(json.dumps({
                'op': 'track',
                'action': 'init',
                'idx': None,
                **PipeManager.ARGS,
                'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
            }))

            while frame_idx < LIMIT:
                start = time.time()
                while frame_idx not in cache:
                    idx, detections, features = inQueue.get()
                    if features is not None:
                        features = torch.from_numpy(features)
                    logger.info(idx)
                    if idx is not None:
                        cache[idx] = (detections, features)
                    else:
                        break
                detections, features = cache[frame_idx]
                logger.info(json.dumps({
                    'op': 'track',
                    'action': 'get',
                    'idx': frame_idx,
                    **PipeManager.ARGS,
                    'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                }))

                start = time.time()
                tracked_objects = tracker.update(torch.from_numpy(detections), [f'{frame_idx}-{d}' for d in range(len(detections))], features)
                del cache[frame_idx]
                logger.info(json.dumps({
                    'op': 'track',
                    'action': 'associate',
                    'idx': frame_idx,
                    **PipeManager.ARGS,
                    'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                }))
                # logger.info(f'{frame_idx:04} update: {time.time() - start}')

                start = time.time()
                outQueue.put((frame_idx, tracked_objects))
                logger.info(json.dumps({
                    'op': 'track',
                    'action': 'put',
                    'idx': frame_idx,
                    **PipeManager.ARGS,
                    'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                }))
                frame_idx += 1

        for _ in range(30):
            outQueue.put((None, None))
        
        # logger.info('done')


class Detect(OpNode):
    def __init__(self, config: str, device: str):
        self.config = config
        self.device = device
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('detect')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-detect-{self.device}.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        try:
            with torch.no_grad():
                start = time.time()
                with open(self.config) as fp1:
                    config = json.load(fp1)
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(config['config']))
                cfg.MODEL.WEIGHTS = config['weights']
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
                cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
                cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
                cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
                cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
                cfg.MODEL.DEVICE = self.device
                predictor = DefaultPredictor(cfg)
                # logger.info(f'init: {time.time() - start}')
                logger.info(json.dumps({
                    'op': 'detect',
                    'action': 'init',
                    'idx': None,
                    'device': self.device,
                    **PipeManager.ARGS,
                    'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                }))

                while True:
                    start = time.time()
                    # idx, frame_masked, frame = inQueue.get()
                    idx, frame_masked_shape, frame_shape = inQueue.get()
                    frame_masked = np.memmap(f'/dev/shm/chanwutk/{idx}-masked.npy', dtype='uint8', mode='r+', shape=frame_masked_shape)
                    frame = np.memmap(f'/dev/shm/chanwutk/{idx}.npy', dtype='uint8', mode='r+', shape=frame_shape)
                    
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'get',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                    if idx is None:
                        break

                    start = time.time()
                    image_regions = regionize_image(frame_masked)
                    # logger.info(f'{idx:04} reorganize: {time.time() - start}')
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'regionize',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    bboxes = []  # xyxy
                    scores = []
                    for _image, _offset in image_regions:
                        _outputs = predictor(_image)
                        _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
                        bboxes += _bboxes
                        scores += _scores
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'predict',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    nms_threshold = config['nms_threshold']
                    nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
                    detections = np.zeros((len(nms_bboxes), 5))
                    detections[:, 0:4] = nms_bboxes
                    detections[:, 4] = nms_scores

                    _detections = np.zeros((len(detections), 6))
                    _detections[:, 0:4] = detections[:, 0:4]
                    _detections[:, 5] = nms_scores
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'nms',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    # logger.info(f'{idx:04}     detect: {time.time() - start}')

                    start = time.time()
                    outQueue.put((idx, _detections, frame))
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'put',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                
        except Exception as e:
            logger.info(str(e))
        
        for _ in range(30):
            outQueue.put((None, None, None))
        # outQueue.put((None, None, None))
        # logger.info('done')


class DecodeMaskDetect(OpNode):
    def __init__(self, filename: str, mask: ElementTree.Element, config: str, device: str):
        self.filename = filename
        self.mask = mask
        self.config = config
        self.device = device

    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('decode-mask-detect')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-decode-mask-detect.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        with torch.no_grad():
            try:
                start = time.time()
                with open(self.config) as fp1:
                    config = json.load(fp1)
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(config['config']))
                cfg.MODEL.WEIGHTS = config['weights']
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
                cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
                cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
                cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
                cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
                cfg.MODEL.DEVICE = self.device
                predictor = DefaultPredictor(cfg)

                cap = cv2.VideoCapture(self.filename)

                # logger.info(f'init: {time.time() - start}')
                logger.info(json.dumps({
                    'op': 'detect',
                    'action': 'init',
                    'idx': None,
                    'device': self.device,
                    **PipeManager.ARGS,
                    'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                }))

                # idx = 0
                decoded_idx = -1
                bitmap, tl, br = None, None, None
                while True:
                    start = time.time()
                    idx = inQueue.get()

                    if idx is None:
                        break
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'get',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    while cap.isOpened() and idx < LIMIT and decoded_idx < idx:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        decoded_idx += 1
                    logger.info(json.dumps({
                        'op': 'detect',
                        'action': 'read',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    if bitmap is None or tl is None or br is None:
                        bitmap, tl, br = get_bitmap(frame, self.mask)
                        bitmap = torch.from_numpy(bitmap).to(self.device)
                    frame_masked = torch.from_numpy(frame).to(self.device)[tl[0]:br[0], tl[1]:br[1], :]
                    frame_masked = bitmap * frame_masked
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'mask',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    outQueue.put((idx, frame_masked.detach().cpu().numpy(), frame))
                    # outQueue.put((idx, frame))
                    # idx += 1
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'put',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                
                # logger.info('done1')
                cap.release()
                cv2.destroyAllWindows() 
                for _ in range(30):
                    # outQueue.put((None, None, None))
                    outQueue.put((None, None))
            except Exception as e:
                logger.info(str(e))
            
            # logger.info('done')
    

class Feature(OpNode):
    def __init__(self, device: str):
        self.device = device
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('feature')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-feature-{self.device}.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        try:
            fe = FeatureExtractor(WEIGHTS / 'osnet_x0_25_msmt17.pt', self.device, True)

            with torch.no_grad():
                while True:
                    start = time.time()
                    idx, detections, frame = inQueue.get()
                    logger.info(json.dumps({
                        'op': 'feature',
                        'action': 'get',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                    if idx is None:
                        break

                    start = time.time()
                    features = fe.features(detections[:, 0:4], frame)
                    if isinstance(features, torch.Tensor):
                        features = features.cpu().numpy()
                    # logger.info(f'{idx:04} feature: {time.time() - start}')
                    logger.info(json.dumps({
                        'op': 'feature',
                        'action': 'feature',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    outQueue.put((idx, detections, features))
                    logger.info(json.dumps({
                        'op': 'feature',
                        'action': 'put',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                
        except Exception as e:
            logger.info(str(e))
        
        for _ in range(30):
            outQueue.put((None, None, None))

        # logger.info('done')


def binpack(inQueue: mp.Queue, outQueue: mp.Queue):
    pass


class Decode(OpNode):
    def __init__(self, filename: str, mask: ElementTree.Element):
        self.filename = filename
        self.mask = mask
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('decode')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-decode.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        with torch.no_grad():
            try:
                cap = cv2.VideoCapture(self.filename)

                idx = 0
                masker = None
                while cap.isOpened() and idx < LIMIT:
                    start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'read',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    # start = time.time()
                    # if masker is None:
                    #     masker = mask_frame(frame, self.mask)
                    # frame_masked = masker(frame)
                    # logger.info(json.dumps({
                    #     'op': 'decode',
                    #     'action': 'mask',
                    #     'idx': idx,
                    #     **PipeManager.ARGS,
                    #     'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    # }))

                    start = time.time()
                    # outQueue.put((idx, frame_masked, frame))
                    outQueue.put((idx, frame))
                    idx += 1
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'put',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                
                # logger.info('done1')
                cap.release()
                cv2.destroyAllWindows() 
                for _ in range(30):
                    # outQueue.put((None, None, None))
                    outQueue.put((None, None))
            except Exception as e:
                logger.info(str(e))
            
            # logger.info('done')


class DecodeMask(OpNode):
    def __init__(self, filename: str, mask: ElementTree.Element, device: str):
        self.filename = filename
        self.mask = mask
        self.device = device
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('decode-mask')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-decode-mask.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        with torch.no_grad():
            try:
                cap = cv2.VideoCapture(self.filename)

                idx = 0
                bitmap, tl, br = None, None, None
                frame_shape = None
                while cap.isOpened() and idx < LIMIT:
                    start = time.time()
                    if frame_shape is None:
                        ret, frame = cap.read()
                        mm_frame = np.memmap(f'/dev/shm/chanwutk/{idx}.npy', dtype='uint8', mode='w+', shape=frame.shape)
                        np.copyto(mm_frame, frame)
                        frame_shape = frame.shape
                    else:
                        mm_frame = np.memmap(f'/dev/shm/chanwutk/{idx}.npy', dtype='uint8', mode='w+', shape=frame_shape)
                        ret, frame = cap.read(mm_frame)
                    if not ret:
                        break
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'read',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    if bitmap is None or tl is None or br is None:
                        bitmap, tl, br = get_bitmap(frame, self.mask)
                        bitmap = torch.from_numpy(bitmap).to(self.device)
                    frame_masked = torch.from_numpy(frame).to(self.device)[tl[0]:br[0], tl[1]:br[1], :]
                    frame_masked = bitmap * frame_masked
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'mask',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    start = time.time()
                    frame_masked = frame_masked.detach().cpu().numpy()
                    mm_frame_masked = np.memmap(f'/dev/shm/chanwutk/{idx}-masked.npy', dtype='uint8', mode='w+', shape=frame_masked.shape)
                    np.copyto(mm_frame_masked, frame_masked)
                    # mm_frame = np.memmap(f'/dev/shm/chanwutk/{idx}.npy', dtype='uint8', mode='w+', shape=frame.shape)
                    # np.copyto(mm_frame, frame)
                    outQueue.put((idx, frame_masked.shape, frame.shape))
                    # outQueue.put((idx, frame))
                    idx += 1
                    logger.info(json.dumps({
                        'op': 'decode',
                        'action': 'put',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                
                # logger.info('done1')
                cap.release()
                cv2.destroyAllWindows() 
                for _ in range(30):
                    outQueue.put((None, None, None))
                    # outQueue.put((None, None))
            except Exception as e:
                logger.info(str(e))
            
            # logger.info('done')


class Mask(OpNode):
    def __init__(self, device: str, bitmap: npt.NDArray, tl: tuple[int, int], br: tuple[int, int]):
        self.bitmap = bitmap
        self.device = device
        self.tl = tl
        self.br = br
    
    def __call__(self, inQueue: mp.Queue, outQueue: mp.Queue):
        logger = logging.getLogger('mask')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./output/{PipeManager.PREFIX}-mask-{self.device}.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)


        with torch.no_grad():
            bitmap = torch.from_numpy(self.bitmap).to(self.device)

            try:
                idx = 0
                while idx < LIMIT:
                    start = time.time()
                    idx, frame = inQueue.get()
                    logger.info(json.dumps({
                        'op': 'mask',
                        'action': 'get',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                    if idx is None:
                        break

                    start = time.time()
                    frame_masked = torch.from_numpy(frame).to(self.device)[self.tl[0]:self.br[0], self.tl[1]:self.br[1], :]
                    frame_masked = bitmap * frame_masked
                    logger.info(json.dumps({
                        'op': 'mask',
                        'action': 'mask',
                        'idx': idx,
                        'device': self.device,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))

                    
                    start = time.time()
                    outQueue.put((idx, frame_masked.detach().cpu().numpy(), frame))
                    idx += 1
                    logger.info(json.dumps({
                        'op': 'mask',
                        'action': 'put',
                        'idx': idx,
                        **PipeManager.ARGS,
                        'time': time.time() - start, 'start_time': start - PipeManager.START, 'end_time': time.time() - PipeManager.START,
                    }))
                
                # logger.info('done1')
                for _ in range(30):
                    outQueue.put((None, None, None))
            except Exception as e:
                logger.info(str(e))
            # logger.info('done')


class PipeManager:
    ARGS: dict
    PREFIX: str
    START: float

    def __init__(self):
        self.processes: list[mp.Process] = []
    
    def __enter__(self):
        return self
    
    def pipe(self, target: OpNode):
        outQueue = mp.Queue(maxsize=40)
        def start(inQueue: "mp.Queue | None" = None):
            process = mp.Process(target=target, args=(inQueue, outQueue))
            process.start()
            self.processes.append(process)
            return outQueue
        
        return start

    def delay(self, target: OpNode):
        outQueue = mp.Queue(maxsize=40)
        def start(inQueue: "mp.Queue | None" = None):
            process = mp.Process(target=target, args=(inQueue, outQueue))
            process.start()
            self.processes.append(process)
        
        return start, outQueue
    
    def consolidate(self, inQueues: list[mp.Queue]):
        outQueue1 = mp.Queue(maxsize=40)
        outQueue2 = mp.Queue(maxsize=40)
        def _consolidate(inQueue: mp.Queue, outQueue: mp.Queue):
            while True:
                out = inQueue.get()
                if out[0] is None:
                    break
                outQueue.put(out)
            outQueue.put(None)
        
        def _consolidateNone(inQueue: mp.Queue, outQueue: mp.Queue, lenInQueues: int):
            noneCount = 0
            outLength = None
            while True:
                out = inQueue.get()
                if out is None:
                    noneCount += 1
                else:
                    outLength = len(out)
                if noneCount == lenInQueues:
                    break
                if out is not None:
                    outQueue.put(out)
            assert outLength is not None
            for _ in range(30):
                outQueue.put(tuple([None] * outLength))
        
        for inQueue in inQueues:
            process = mp.Process(target=_consolidate, args=(inQueue, outQueue1))
            process.start()
            self.processes.append(process)
        
        process = mp.Process(target=_consolidateNone, args=(outQueue1, outQueue2, len(inQueues)))
        process.start()
        self.processes.append(process)

        return outQueue2
    
    def close(self):
        for process in self.processes:
            process.join()
            process.terminate()

    def __exit__(self, *args):
        for process in self.processes:
            process.join()
            process.terminate()
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='Example detection and tracking script')
    parser.add_argument('-v', '--video', required=True,
                        help='Input video')
    parser.add_argument('-c', '--config', required=True,
                        help='Detection model configuration')
    parser.add_argument('-m', '--mask', required=True,
                        help='Mask for the video')
    return parser.parse_args()


GPUS = [0, 2, 3, 4, 6]


def main(args):
    tree = ElementTree.parse(args.mask)
    mask = tree.getroot()

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = cap.read()[1]
    cap.release()
    cv2.destroyAllWindows()
    masker = mask_frame(frame, mask)
    bitmap, tl, br = get_bitmap(frame, mask)

    for num_processes in range(3, len(GPUS) + 1):
        with PipeManager() as pm:
            PipeManager.ARGS = {
                'num_processes': num_processes,
            }
            PipeManager.PREFIX = f'parallel-{num_processes}'
            PipeManager.START = time.time()

            start = time.time()
            # startDecode, decode = pm.delay(DecodeMask(os.path.expanduser(args.video), mask, f'cuda:0'))
            decode = pm.pipe(DecodeMask(os.path.expanduser(args.video), mask, f'cuda:{GPUS[0]}'))()
            detects = [pm.pipe(Detect(args.config, f'cuda:{GPUS[i]}'))(decode) for i in range(num_processes)]
            detect = pm.consolidate(detects)
            # features = [pm.pipe(Feature(f'cuda:{GPUS[-i - 1]}'))(detect) for i in range(num_processes)]
            # feature = pm.consolidate(features)
            # track = pm.pipe(Track(width, height))(feature)
            # pm.pipe(Write())(track)
            pm.pipe(Write())(detect)
            # pm.pipe(WriteVideo('output/parallel.mp4'))(mask)

            # time.sleep(10)
            # start = time.time()
            # startDecode(None)
        runtime = time.time() - start
        end = time.time()
        print(f'time: {runtime}')
        with open(f'output/parallel-{num_processes}-done.log', 'w') as fp:
            fp.write(json.dumps({
                'op': 'all',
                'action': 'done',
                'idx': None,
                'num_processes': num_processes,
                'time': runtime,
                'start_time': start,
                'end_time': end,
            }))
        # break


if __name__ == "__main__":
    main(parse_args())
