import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Queue
import threading
import ctypes
import time
import json

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from utils import mask_frame, parse_outputs, regionize_image


def image4k(shm):
    img = np.ndarray((2160, 3840, 3), dtype=np.uint8, buffer=shm.buf)
    img[990:1000, 990:1000, :] = 0
    return img


def image4k2():
    img = np.ndarray((2160, 3840, 3), dtype=np.uint8)
    img[990:1000, 990:1000, :] = 0
    return img


def write_log(log_file, queue):
    with open(log_file, 'w') as fp:
        while True:
            j = queue.get()
            if j is None:
                break
            fp.write(j + '\n')


def main():
    with torch.no_grad():
        logQueue = mp.Queue()
        logProcess = mp.Process(target=write_log, args=('./logs/test_sharedmemory_latency.log', logQueue))
        logProcess.start()

        with open('./configs/config_refined.json') as fp1:
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
        cfg.MODEL.DEVICE = 'cuda:2'
        predictor = DefaultPredictor(cfg)
        # logger.info(f'init: {time.time() - start}')
    
        for _ in range(10):
            data = image4k2()
            frame_masked = data
            image_regions = regionize_image(frame_masked)
            for _image, _offset in image_regions:
                _outputs = predictor(_image)
                _bboxes, _scores, _ = parse_outputs(_outputs, _offset)

        queue1 = mp.Queue()
        _start = time.time_ns()
        for _ in range(50):
            start = time.time_ns()
            data = image4k2()
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '1read',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            queue1.put(data)
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '2put',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))


            start = time.time_ns()
            frame_masked = queue1.get()
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '3get',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            image_regions = regionize_image(frame_masked)
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '4region',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            for _image, _offset in image_regions:
                _outputs = predictor(_image)
                _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '5predict',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '6cleanup',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))
        queue1.close()
        

        queue1 = mp.Queue()
        _start = time.time_ns()
        for _ in range(50):
            start = time.time_ns()
            shm = shared_memory.SharedMemory(create=True, size=3840 * 2160 * 3, name='shared_data')
            data = image4k(shm)
            end = time.time_ns()
            shm.close()
            logQueue.put(json.dumps({
                'action': '1read',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            queue1.put('shared_data')
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '2put',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))


            start = time.time_ns()
            name = queue1.get()
            shm = shared_memory.SharedMemory(name=name)
            data = np.ndarray((3840, 2160, 3), dtype=np.uint8, buffer=shm.buf)
            frame_masked = data
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '3get',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            # frame_masked = np.empty_like(data)
            # # np.copyto(frame_masked, data)
            # frame_masked[:] = data
            image_regions = regionize_image(frame_masked)
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '4region',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            for _image, _offset in image_regions:
                _outputs = predictor(_image)
                _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '5predict',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            start = time.time_ns()
            shm.close()
            shm.unlink()
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '6cleanup',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))
        queue1.close()


        logQueue.put(None)
        logProcess.join()
        logProcess.terminate()
        

if __name__ == '__main__':
    main()