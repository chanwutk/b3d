import json
import multiprocessing as mp
import os
import shutil
from xml.etree import ElementTree

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from external.nms import nms
from external.sort import Sort
from utils import get_mask, parse_outputs, regionize_image


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


def track(videofile: str, gpu: int):
    filename = videofile.split('/')[-1]
    scenario = filename.replace('.mp4', '')

    fl = open(f'./tracking-results/{filename}.log.jsonl', 'w')
    fDet = open(f'./tracking-results/{filename}.det.jsonl', 'w')
    fRender = open(f'./tracking-results/{filename}.rendering.jsonl', 'w')

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
    cfg.MODEL.DEVICE = f'cuda:{gpu}'
    device = torch.device(cfg.MODEL.DEVICE)
    predictor = DefaultPredictor(cfg)

    tree = ElementTree.parse('all-masks.xml')
    mask = tree.getroot()

    mask = mask.find(f'.//image[@name="{filename.replace('.mp4', '.jpg')}"]')
    assert isinstance(mask, ElementTree.Element), (mask, type(mask))

    tracker = Sort(max_age=5)
    cap = cv2.VideoCapture(videofile)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_interval = num_frames // 5
    trajectories = {}
    rendering = {}
    frame_index = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_bitmap, mtl, mbr = get_mask(mask, width, height)
    mtltl = np.array([*mtl, *mtl])
    mask_bitmap = torch.from_numpy(mask_bitmap).to(device)
    while cap.isOpened():
        fl.write('Parsing frame {:d} / {:d}...\n'.format(frame_index, frame_count))
        if frame_index % 20 == 0:
            fl.flush()
        success, frame = cap.read()
        if not success:
            break
        # frame_masked = mask_frame(frame, mask)
        cropped_frame = frame[mtl[0]:mbr[0], mtl[1]:mbr[1], :]
        frame_masked = mask_bitmap * torch.from_numpy(cropped_frame).to(device)
        frame_masked = frame_masked.detach().cpu().numpy()

        image_regions = regionize_image(frame_masked)
        bboxes = []
        scores = []
        for _image, _offset in image_regions:
            _outputs = predictor(_image)
            _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
            bboxes += _bboxes
            scores += _scores
        nms_threshold = config['nms_threshold']
        nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
        detections = np.zeros((len(nms_bboxes), 5))
        if len(nms_bboxes) != 0:
            detections[:, 0:4] = nms_bboxes
            detections[:, 4] = nms_scores
        fDet.write(json.dumps([frame_index, detections.tolist()]) + '\n')
        
        tracked_objects = tracker.update(detections)
        rendering[frame_index] = []
        for tracked_object in tracked_objects:
            tl = (int(tracked_object[0]), int(tracked_object[1]))
            br = (int(tracked_object[2]), int(tracked_object[3]))
            object_index = int(tracked_object[4])
            if object_index not in trajectories:
                trajectories[object_index] = []
            trajectories[object_index].append([
                frame_index, tl[0], tl[1], br[0], br[1]])
            rendering[frame_index].append([
                object_index, tl[0], tl[1], br[0], br[1]])

        if frame_index % image_interval == 0:
            cv2.imwrite(f'tracking-results/{filename}.{frame_index}.jpg', frame_masked)

            frame_vis = frame.copy().astype(np.uint16) * 2
            frame_vis[mtl[0]:mbr[0], mtl[1]:mbr[1], :] += frame_masked.astype(np.uint16) * 3
            frame_vis //= 5
            frame_vis = frame_vis.astype(np.uint8)

            frame_vis1 = frame_vis.copy()
            for bbox in detections:
                bbox = bbox.copy()
                bbox[:4] += mtltl[::-1]
                tl = (int(bbox[0]), int(bbox[1]))
                br = (int(bbox[2]), int(bbox[3]))
                frame_vis1 = cv2.rectangle(frame_vis1, tl, br, (0, 255, 0), 2)
            cv2.imwrite(f'tracking-results/{filename}.{frame_index}.det.jpg', frame_vis1)

            frame_vis2 = frame_vis.copy()
            for tracked_object in tracked_objects:
                tracked_object = tracked_object.copy()
                tracked_object[:4] += mtltl[::-1]
                tl = (int(tracked_object[0]), int(tracked_object[1]))
                br = (int(tracked_object[2]), int(tracked_object[3]))
                object_index = int(tracked_object[4])
                frame_vis2 = cv2.rectangle(frame_vis2, tl, br, colors[object_index % len(colors)], 2)
            cv2.imwrite(f'tracking-results/{filename}.{frame_index}.trk.jpg', frame_vis2)
        
        fRender.write(json.dumps([frame_index, rendering[frame_index]]) + '\n')

        frame_index = frame_index + 1
    cap.release()
    cv2.destroyAllWindows()

    with open('tracking-results/{}.trajectories.json'.format(scenario), 'w') as fp:
        json.dump(trajectories, fp)
    
    fl.close()
    fDet.close()
    fRender.close()


BASE_DIR = '/work/chanwutk/fangyu-drone/fangyu'


if __name__ == '__main__':
    if os.path.exists('tracking-results'):
        shutil.rmtree('tracking-results')
    os.makedirs('tracking-results')

    processes = []
    for i, videofile in enumerate(os.listdir(BASE_DIR)):
        args = os.path.join(BASE_DIR, videofile), (i % torch.cuda.device_count())
        process = mp.Process(target=track, args=args)
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    for process in processes:
        process.terminate()
