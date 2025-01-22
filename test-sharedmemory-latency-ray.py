import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Queue
import threading
import ctypes
import time
import json
from PIL import Image
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from utils import mask_frame, parse_outputs, regionize_image

import queue

import ray
import ray.util.queue as rq




cap = cv2.VideoCapture('jnc00.mp4')
frame = None
for i in range(20):
    ret, _frame = cap.read()
    if not ret:
        break
    frame = _frame

cap.release()


def image4k(shm):
    img = np.ndarray((2160, 3840, 3), dtype=np.uint8, buffer=shm.buf)
    img[:] = frame
    img[990:1000, 990:1000, :] = 0
    return img


def image4k2():
    img = np.ndarray((2160, 3840, 3), dtype=np.uint8)
    img[:] = frame
    img[990:1000, 990:1000, :] = 0
    return img


def write_log(log_file, queue):
    with open(log_file, 'w') as fp:
        fp.write('[\n')
        first = True
        while True:
            j = queue.get()
            if j is None:
                break
            else:
                if first:
                    first = False
                else:
                    fp.write(',\n')
            fp.write(j)
        fp.write(']\n')


def apply_image(transform, img, interp=None, device="cpu"):
    # assert img.shape[:2] == (transform.h, transform.w)
    # assert len(img.shape) <= 4
    interp_method = interp if interp is not None else transform.interp

    # if any(x < 0 for x in img.strides):
    #     img = np.ascontiguousarray(img)

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    img = img.to(device).to(torch.float32)

    shape = list(img.shape)
    shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
    img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
    _PIL_RESIZE_TO_INTERPOLATE_MODE = {
        Image.NEAREST: "nearest",
        Image.BILINEAR: "bilinear",
        Image.BICUBIC: "bicubic",
    }
    mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
    align_corners = None if mode == "nearest" else False
    img = F.interpolate(
        img, (transform.new_h, transform.new_w), mode=mode, align_corners=align_corners
    )
    shape[:2] = (transform.new_h, transform.new_w)
    ret = img.permute(2, 3, 0, 1).view(shape).cpu().numpy()  # nchw -> hw(c)
    
    return ret


def transform_image(model, original_image):
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if model.input_format == "RGB":
            print('RGB')
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        transform = model.aug.get_transform(original_image)

        image = apply_image(transform, original_image, device=model.cfg.MODEL.DEVICE)

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(model.cfg.MODEL.DEVICE)

        inputs = {"image": image, "height": height, "width": width}
        return inputs


def predict(model, input):
    with torch.no_grad():
        predictions = model.model([input])[0]
        return predictions


@ray.remote
class ReadImage:
    def __init__(self, file, expr, start_queue: rq.Queue):
        self.queue = rq.Queue(maxsize=10)
        # self.queue = Queue()
        self.file = file
        self.expr = expr
        self.start_queue = start_queue

    def get(self):
        return self.queue.get()
    
    async def run(self):
        with open(f'./logs/test_read_image_{self.expr}.log', 'w') as fp:
            log = make_log2(fp, self.expr)
            idx = 0
            cap = cv2.VideoCapture(self.file)
            print('waiting for start')
            start_value = self.start_queue.get()
            print('start received')
            self.queue.put(start_value)
            while idx < 30:
                start = time.time_ns()
                ret, frame = cap.read()
                end = time.time_ns()
                log('read', start, end)

                if not ret:
                    break

                start = time.time_ns()
                # yield idx, frame
                self.queue.put((idx, frame))
                end = time.time_ns()
                log('put', start, end)
                time.sleep(0.5)

                idx += 1
            cap.release()
            # yield None, None
            self.queue.put((None, None))


@ray.remote(num_returns="dynamic", num_cpus=2)
def read_image(file, expr, start_queue: rq.Queue):
    with open(f'./logs/test_read_image_{expr}.log', 'w') as fp:
        log = make_log2(fp, expr)
        idx = 0
        cap = cv2.VideoCapture(file)
        start_value = start_queue.get()
        yield start_value
        while idx < 30:
            start = time.time_ns()
            ret, frame = cap.read()
            end = time.time_ns()
            log('read', start, end)

            if not ret:
                break

            start = time.time_ns()
            yield idx, frame
            end = time.time_ns()
            log('put', start, end)

            idx += 1
        cap.release()
        yield None, None


@ray.remote
class ReadImageSharedMemory:
    def __init__(self, file, expr, start_queue: rq.Queue):
        self.queue = rq.Queue(maxsize=10)
        # self.queue = Queue()
        self.file = file
        self.expr = expr
        self.start_queue = start_queue
    
    def get(self):
        return self.queue.get()

    async def run(self):
        with open(f'./logs/test_read_image_shared_{self.expr}.log', 'w') as fp:
            log = make_log2(fp, self.expr)
            idx = 0
            cap = cv2.VideoCapture(self.file)
            print('waiting for start')
            start_value = self.start_queue.get()
            print('start received')
            self.queue.put(start_value)
            while idx < 30:
                start = time.time_ns()
                try:
                    shm = shared_memory.SharedMemory(create=True, size=3840 * 2160 * 3, name='sharedimg_' + str(idx))
                except FileExistsError:
                    shm = shared_memory.SharedMemory(create=False, size=3840 * 2160 * 3, name='sharedimg_' + str(idx))
                img = np.ndarray((2160, 3840, 3), dtype=np.uint8, buffer=shm.buf)
                ret, _ = cap.read(img)
                shm.close()
                end = time.time_ns()
                log('read', start, end)
                if not ret:
                    break
                start = time.time_ns()
                self.queue.put((idx, None))
                # yield idx, None
                end = time.time_ns()
                log('put', start, end)
                time.sleep(0.5)

                idx += 1
            cap.release()
            self.queue.put((None, None))
            # yield None, None


@ray.remote(num_cpus=2, num_returns="dynamic")
def read_image_shared_memory(file, expr, start_queue: rq.Queue):
    with open(f'./logs/test_read_image_shared_{expr}.log', 'w') as fp:
        log = make_log2(fp, expr)
        idx = 0
        cap = cv2.VideoCapture(file)
        start_value = start_queue.get()
        yield start_value
        while idx < 30:
            start = time.time_ns()
            try:
                shm = shared_memory.SharedMemory(create=True, size=3840 * 2160 * 3, name='sharedimg_' + str(idx))
            except FileExistsError:
                shm = shared_memory.SharedMemory(create=False, size=3840 * 2160 * 3, name='sharedimg_' + str(idx))
            img = np.ndarray((2160, 3840, 3), dtype=np.uint8, buffer=shm.buf)
            ret, _ = cap.read(img)
            shm.close()
            end = time.time_ns()
            log('read', start, end)
            if not ret:
                break
            start = time.time_ns()
            # self.queue.put((idx, None))
            yield idx, None
            end = time.time_ns()
            log('put', start, end)

            idx += 1
        cap.release()
        # self.queue.put((None, None))
        yield None, None


def make_log(logQueue, expr):
    def write_log(action, start, end):
        print(action)
        logQueue.put(json.dumps({
            'action': action,
            'expr': expr,
            'start': start,
            'end': end,
            'elapsed': end - start,
        }))
    return write_log


def make_log2(f, expr):
    def write_log(action, start, end):
        print(action)
        f.write(json.dumps({
            'action': action,
            'expr': expr,
            'start': start,
            'end': end,
            'elapsed': end - start,
        }) + '\n')
        f.flush()
    return write_log


@ray.remote(num_gpus=1, num_cpus=2)
def worker_with_pipelining(work_queue, expr, start_queue: rq.Queue):
    with open(f'./logs/test_worker_{expr}.log', 'w') as fp:
        log = make_log2(fp, expr)

        with torch.no_grad():
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
            cfg.MODEL.DEVICE = 'cuda:0'
            print('create model')
            predictor = DefaultPredictor(cfg)
            print('model created')

            cap = cv2.VideoCapture('jnc00.mp4')
            for _ in range(2):
                _, data = cap.read()
                frame_masked = data
                image_regions = regionize_image(frame_masked)
                for _image, _offset in image_regions:
                    # _outputs = predictor(_image)
                    _input = transform_image(predictor, torch.from_numpy(_image))
                    _outputs = predict(predictor, _input)
                    _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
            cap.release()
            
            print('warmed up')

            start_queue.put('start')
            # iterator_ref = work_queue

            start = time.time_ns()
            # work_item_ref = next(iterator_ref)
            work_item_ref = work_queue.get.remote()
            end = time.time_ns()
            log('get_work_ref', start, end)


            idx = None
            while True:
            # for work_item_ref in iterator_ref:
                # Get work from the remote queue.
                start = time.time_ns()
                work_item = ray.get(work_item_ref)
                end = time.time_ns()
                log('get_work_item', start, end)

                if work_item is None:
                    print('work_item is None --------------------------------------------------------------', idx)
                    break

                start = time.time_ns()
                # work_item_ref = next(iterator_ref)
                work_item_ref = work_queue.get.remote()
                end = time.time_ns()
                log('get_work_ref', start, end)
                if work_item == 'start':
                    continue

                # Do work while we are fetching the next work item.
                start = time.time_ns()
                idx, frame = work_item
                if idx is None:
                    print('idx is None --------------------------------------------------------------')
                    break
                
                shm = None
                if frame is None:
                    shm = shared_memory.SharedMemory(name='sharedimg_' + str(idx))
                    frame = torch.frombuffer(shm.buf, dtype=torch.uint8).reshape((2160, 3840, 3))
                end = time.time_ns()
                log('get_frame', start, end)

                start = time.time_ns()
                _input = transform_image(predictor, frame)
                end = time.time_ns()
                log('transform', start, end)


                start = time.time_ns()
                _outputs = predict(predictor, _input)
                end = time.time_ns()
                log('predict', start, end)


                start = time.time_ns()
                parse_outputs(_outputs, (0, 0))
                end = time.time_ns()
                log('parse', start, end)

                start = time.time_ns()
                if shm is not None:
                    shm.close()
                    shm.unlink()
                end = time.time_ns()
                log('cleanup', start, end)




# https://docs.ray.io/en/latest/ray-core/patterns/pipelining.html


def main():
    # with torch.no_grad():
    #     logQueue = mp.Queue()
    #     logProcess = mp.Process(target=write_log, args=('./logs/test_sharedmemory_latency.log', logQueue))
    #     logProcess.start()

    #     with open('./configs/config_refined.json') as fp1:
    #         config = json.load(fp1)
    #     cfg = get_cfg()
    #     cfg.merge_from_file(model_zoo.get_config_file(config['config']))
    #     cfg.MODEL.WEIGHTS = config['weights']
    #     cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
    #     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
    #     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
    #     cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    #     cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
    #     cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
    #     cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
    #     cfg.MODEL.DEVICE = 'cuda:2'
    #     predictor = DefaultPredictor(cfg)
    #     # logger.info(f'init: {time.time_ns() - start}')
    
    #     q = Queue()
    #     for _ in range(10):
    #         data = image4k2()
    #         frame_masked = data
    #         image_regions = regionize_image(frame_masked)
    #         for _image, _offset in image_regions:
    #             # _outputs = predictor(_image)
    #             _input = transform_image(predictor, torch.from_numpy(_image))
    #             _outputs = predict(predictor, _input)
    #             _bboxes, _scores, _ = parse_outputs(_outputs, _offset)

    #     log = make_log(logQueue, 'copy')
    #     queue1 = mp.Queue()
    #     cap = cv2.VideoCapture('jnc00.mp4')
    #     _start = time.time_ns()
    #     for _ in range(30):
    #         start = time.time_ns()
    #         ret, frame = cap.read()
    #         end = time.time_ns()
    #         log('read', start, end)

    #         assert ret is not None

    #         start = time.time_ns()
    #         queue1.put(frame)
    #         end = time.time_ns()
    #         log('put', start, end)


    #         start = time.time_ns()
    #         frame_masked = queue1.get()
    #         end = time.time_ns()
    #         log('get_work_item', start, end)

    #         start = time.time_ns()
    #         _input = transform_image(predictor, _image)
    #         end = time.time_ns()
    #         log('transform', start, end)

    #         start = time.time_ns()
    #         _outputs = predict(predictor, _input)
    #         end = time.time_ns()
    #         log('predict', start, end)


    #         start = time.time_ns()
    #         parse_outputs(_outputs, _offset)
    #         end = time.time_ns()
    #         log('parse', start, end)

    #         start = time.time_ns()
    #         end = time.time_ns()
    #         log('cleanup', start, end)
    #     queue1.close()
        

    #     log = make_log(logQueue, 'shared_memory')
    #     queue1 = mp.Queue()
    #     cap = cv2.VideoCapture('jnc00.mp4')
    #     _start = time.time_ns()
    #     for idx in range(30):
    #         start = time.time_ns()
    #         shm = shared_memory.SharedMemory(create=True, size=3840 * 2160 * 3, name='shareddata')
    #         img = np.ndarray((2160, 3840, 3), dtype=np.uint8, buffer=shm.buf)
    #         ret, _ = cap.read(img)
    #         shm.close()
    #         end = time.time_ns()
    #         log('read', start, end)

    #         start = time.time_ns()
    #         queue1.put('shareddata')
    #         end = time.time_ns()
    #         log('put', start, end)


    #         start = time.time_ns()
    #         name = queue1.get()
    #         shm = shared_memory.SharedMemory(name=name)
    #         # data = np.ndarray((3840, 2160, 3), dtype=np.uint8, buffer=shm.buf)
    #         data = torch.frombuffer(shm.buf, dtype=torch.uint8).reshape((2160, 3840, 3))
    #         end = time.time_ns()
    #         log('get_work_item', start, end)

    #         start = time.time_ns()
    #         _input = transform_image(predictor, _image)
    #         end = time.time_ns()
    #         log('transform', start, end)

    #         start = time.time_ns()
    #         _outputs = predict(predictor, _input)
    #         end = time.time_ns()
    #         log('predict', start, end)

    #         start = time.time_ns()
    #         parse_outputs(_outputs, _offset)
    #         end = time.time_ns()
    #         log('parse', start, end)

    #         start = time.time_ns()
    #         shm.close()
    #         shm.unlink()
    #         end = time.time_ns()
    #         log('cleanup', start, end)
    #     queue1.close()


    #     logQueue.put(None)
    #     logProcess.join()
    #     logProcess.terminate()
        

    start_queue = rq.Queue(maxsize=1)
    # work_queue = read_image_shared_memory.remote('jnc00.mp4', 'ray-shared-memory', start_queue)
    work_queue = ReadImageSharedMemory.remote('jnc00.mp4', 'ray-shared-memory', start_queue)
    worker = worker_with_pipelining.remote(work_queue, 'ray-shared-memory', start_queue)
    work_queue.run.remote()
    ray.get(worker)

    start_queue = rq.Queue(maxsize=1)
    # work_queue = read_image.remote('jnc00.mp4', 'ray-shared-memory', start_queue)
    work_queue = ReadImage.remote('jnc00.mp4', 'ray-shared-memory', start_queue)
    worker = worker_with_pipelining.remote(work_queue, 'ray-shared-memory', start_queue)
    work_queue.run.remote()
    ray.get(worker)


if __name__ == '__main__':
    main()