import argparse
import json
import time
import multiprocessing as mp
from xml.etree import ElementTree
import time
import logging
import os

import cv2
import numpy as np
from matplotlib.path import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

from external.nms import nms
from utils import parse_outputs, regionize_image


LOG_DIR = './logs'
RESULTS_DIR = './results'
LIMIT = 3000


def detector(in_queue, out_queue, args):
    gpu_id, = args

    # create logger with 'reader'
    logger = logging.getLogger(f'detector_{gpu_id}')
    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(LOG_DIR, f'detector_{gpu_id}.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    start = time.time()

    with open('./configs/config_refined.json') as fp:
        config = json.load(fp)
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
    # if gpu_id >= torch.cuda.device_count():
    #     cfg.MODEL.DEVICE = 'cpu'
    # else:
    #     cfg.MODEL.DEVICE = f'cuda:{gpu_id}'

    cfg.MODEL.DEVICE = f'cuda:{gpu_id}'
    # cfg.MODEL.DEVICE = f'cuda:0'
    predictor = DefaultPredictor(cfg)

    logger.info(json.dumps({'action': 'init', 'runtime': time.time() - start}))

    while True:
        start = time.time()
        frame_offset_idx = in_queue.get()
        if frame_offset_idx is None:
            break

        frame, _offset, idx, len_image, masktl, region_idx = frame_offset_idx
        logger.info(json.dumps({'action': 'get_image', 'runtime': time.time() - start, 'offset': _offset, 'masktl': masktl, 'region_idx': region_idx, 'regiou_len': len_image, 'frame_idx': idx}))
        start = time.time()
        _outputs = predictor(frame)
        _offset = (_offset[0] + masktl[1], _offset[1] + masktl[0])
        _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
        logger.info(json.dumps({
            'action': 'prediction', 'runtime': time.time() - start, 'offset': _offset, 'masktl': masktl, 'region_idx': region_idx, 'regiou_len': len_image, 'frame_idx': idx,
            'bboxes': [b.tolist() for b in _bboxes], 'scores': [s.tolist() for s in _scores]
        }))

        out_queue.put((_bboxes, _scores, idx, len_image))
    
    out_queue.put(None)


def get_mask(w, h, mask):
    domain = mask.find('.//polygon[@label="domain"]').attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    width, height = int(w), int(h)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width))
    # frame[bitmap == 0] = 0
    # frame_masked = frame[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap.astype(np.uint8).reshape((height, width, 1)), domain_poly, tl, br


def reader(filename, in_queue):
    # create logger with 'reader'
    logger = logging.getLogger('reader')
    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(LOG_DIR, 'reader.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    tree = ElementTree.parse(filename + '.mask.xml')
    mask = tree.getroot()

    cap = cv2.VideoCapture(filename)
    frame_index = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_array = None
    while cap.isOpened():
        if frame_index >= LIMIT:
            break
        # print(f'{frame_index}/{frame_count}')
        start = time.time()
        success, frame = cap.read()
        if not success:
            break

        if mask_array is None:
            mask_array, domain_poly, masktl, maskbr = get_mask(frame.shape[1], frame.shape[0], mask)
            mask_array = mask_array[masktl[0]:maskbr[0], masktl[1]:maskbr[1], :]

            def apply_mask(frame):
                return frame[masktl[0]:maskbr[0], masktl[1]:maskbr[1], :] * mask_array
        frame_masked = apply_mask(frame)

        image_regions = regionize_image(frame_masked)
        logger.info(json.dumps({'frame_index': frame_index, 'time': time.time() - start, 'num_regions': len(image_regions)}))
        for region_idx, (_image, _offset) in enumerate(image_regions):
            in_queue.put((_image, _offset, frame_index, len(image_regions), masktl, region_idx), block=True, timeout=None)
        frame_index += 1
    
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)
    in_queue.put(None)


def pipe(in_queue, out_queue):
    while True:
        data = in_queue.get()
        if data is None:
            break
        out_queue.put(data)
    out_queue.put(None)


def main(args: argparse.Namespace):
    input_file = args.input
    output_file = args.output
    with open('./configs/config_refined.json') as fp:
        config = json.load(fp)

    num_processes = 6
    # num_processes = torch.cuda.device_count()
    # num_processes = 20

    cap = cv2.VideoCapture(input_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        in_queue = mp.Queue(20)
        out_queue = mp.Queue()

        pDetectors = []
        for i in range(num_processes):
            pDetector = mp.Process(target=detector, args=(in_queue, out_queue, (i,)))
            pDetector.start()
            pDetectors.append(pDetector)
        
        pReader = mp.Process(target=reader, args=(input_file, in_queue))
        pReader.start()

        bboxes = {}
        none_count = 0

        writer = open(output_file or f'{input_file}.detections.jsonl', 'w')
        start_time = time.time()
        completed_detection_count = 0
        pbar = tqdm(total=frame_count)
        while none_count < num_processes:
            bboxes_scores_idx = out_queue.get()
            if bboxes_scores_idx is None:
                none_count += 1
                continue
            _bboxes, _scores, idx, len_image = bboxes_scores_idx

            if idx not in bboxes:
                bboxes[idx] = []
            bboxes[idx].append((_bboxes, _scores))
            if len(bboxes[idx]) == len_image:
                pbar.update(1)
                # b, s = [], []
                # for _bboxes, _scores in bboxes[idx]:
                #     b += _bboxes
                #     s += _scores

                # nms_threshold = config['nms_threshold']
                # nms_bboxes, nms_scores = nms(b, s, nms_threshold)
                # detections = np.zeros((len(nms_bboxes), 5))
                # detections[:, 0:4] = nms_bboxes
                # detections[:, 4] = nms_scores

                # writer.write("[" + str(idx) + ", " + json.dumps(detections.tolist()) + "]\n")
                # print(f'detection speed:', int(completed_detection_count * 100 / (time.time() - start_time)) / 100., 'fps')
                completed_detection_count += 1
        
        pbar.close()
        writer.close()
        
        pReader.join()
        pReader.terminate()
        for pDetector in pDetectors:
            pDetector.join()
            pDetector.terminate()
    except KeyboardInterrupt:
        try: pReader.terminate()
        except: pass

        try:
            for pDetector in pDetectors:
                pDetector.join()
                pDetector.terminate()
        except: pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input video file', default='./hwy00.mp4')
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())
