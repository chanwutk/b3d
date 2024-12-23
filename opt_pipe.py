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

def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    return tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
colors: list[tuple[int, int, int]] = [hex_to_rgb(c) for c in colors_]


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
    bitmap = bitmap.reshape((height, width))
    def masker(frame: npt.NDArray):
        frame[bitmap == 0] = 0
        frame_masked = frame[tl[0]:br[0], tl[1]:br[1], :]
        return frame_masked
    setattr(masker, 't', tl[0])
    setattr(masker, 'l', tl[1])
    # setattr(masker, 'tl', tl)
    # setattr(masker, 'br', br)
    return masker


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


def main(args):
    with open(args.config) as fp:
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
    cfg.MODEL.DEVICE = f'cuda:0'
    predictor = DefaultPredictor(cfg)
    tree = ElementTree.parse(args.mask)
    mask = tree.getroot()

    tracker = Sort(max_age=5)
    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    trajectories = {}
    rendering = {}
    frame_index = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    masker = None
    start = time.time()
    writer = cv2.VideoWriter(
        f'output/opt_none-{os.path.expanduser(args.video)}',
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)
    )
    while cap.isOpened() and frame_index < 3000:
        print('Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
        success, frame = cap.read()
        if not success:
            break
        if masker is None:
            masker = mask_frame(frame, mask)
        frame_masked = masker(frame)

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
        detections[:, 0:4] = nms_bboxes
        detections[:, 4] = nms_scores

        tracked_objects = tracker.update(detections)
        for tracked_object in tracked_objects:
            tl = (int(tracked_object[0]) + masker.l, int(tracked_object[1]) + masker.t)
            br = (int(tracked_object[2]) + masker.l, int(tracked_object[3]) + masker.t)
            object_index = int(tracked_object[4])
            
            frame = cv2.rectangle(frame, tl, br, colors[object_index % len(colors)], 2)
        
        writer.write(frame)

        frame_index = frame_index + 1
    
    cap.release()
    cv2.destroyAllWindows()

    end = time.time()
    with open(f'output/opt_none-{os.path.expanduser(args.video)}.time', 'w') as fp:
        fp.write(f'{end - start}')


if __name__ == '__main__':
    main(parse_args())
