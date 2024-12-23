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


def apply_image(transform, img, logQueue, expr, _start, interp=None, device="cpu"):
    start = time.time_ns()
    # assert img.shape[:2] == (transform.h, transform.w)
    # assert len(img.shape) <= 4
    interp_method = interp if interp is not None else transform.interp

    # if any(x < 0 for x in img.strides):
    #     img = np.ascontiguousarray(img)
    end = time.time_ns()
    logQueue.put(json.dumps({
        'action': '6config',
        'expr': expr,
        'start': start - _start,
        'end': end - _start,
        'time': end - start
    }))

    start = time.time_ns()
    # img = torch.from_numpy(img).to(device).to(torch.float32)
    img = img.to(device).to(torch.float32)
    end = time.time_ns()
    logQueue.put(json.dumps({
        'action': '7togpu',
        'expr': expr,
        'start': start - _start,
        'end': end - _start,
        'time': end - start
    }))

    start = time.time_ns()
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
    end = time.time_ns()
    logQueue.put(json.dumps({
        'action': '8applytransform',
        'expr': expr,
        'start': start - _start,
        'end': end - _start,
        'time': end - start
    }))
    
    return ret


def transform_image(model, original_image, logQueue, expr, _start):
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        start = time.time_ns()
        if model.input_format == "RGB":
            print('RGB')
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        transform = model.aug.get_transform(original_image)
        end = time.time_ns()
        logQueue.put(json.dumps({
            'action': '5gettransform',
            'expr': expr,
            'start': start - _start,
            'end': end - _start,
            'time': end - start
        }))

        # start = time.time_ns()
        image = apply_image(transform, original_image, logQueue, expr, _start, device=model.cfg.MODEL.DEVICE)
        # end = time.time_ns()
        # logQueue.put(json.dumps({
        #     'action': '6applyimage',
        #     'expr': expr,
        #     'start': start - _start,
        #     'end': end - _start,
        #     'time': end - start
        # }))

        # print(image.dtype, image.shape, image.device, model.cfg.MODEL.DEVICE)
        start = time.time_ns()
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(model.cfg.MODEL.DEVICE)
        end = time.time_ns()
        logQueue.put(json.dumps({
            'action': '9astensor',
            'expr': expr,
            'start': start - _start,
            'end': end - _start,
            'time': end - start
        }))

        inputs = {"image": image, "height": height, "width": width}
        return inputs


def predict(model, input):
    with torch.no_grad():
        predictions = model.model([input])[0]
        return predictions


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
    
        q = Queue()
        for _ in range(10):
            data = image4k2()
            frame_masked = data
            image_regions = regionize_image(frame_masked)
            for _image, _offset in image_regions:
                # _outputs = predictor(_image)
                _input = transform_image(predictor, torch.from_numpy(_image), q, '', 0)
                _outputs = predict(predictor, _input)
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
            # image_regions = regionize_image(frame_masked)
            image_regions = [(torch.from_numpy(frame_masked), (0, 0))]
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '4region',
                'expr': 'copy',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            for _image, _offset in image_regions:
                # _outputs = predictor(_image)

                # start = time.time_ns()
                _input = transform_image(predictor, _image, logQueue, 'copy', _start)
                # end = time.time_ns()
                # logQueue.put(json.dumps({
                #     'action': '5transform',
                #     'expr': 'copy',
                #     'start': start - _start,
                #     'end': end - _start,
                #     'time': end - start
                # }))

                start = time.time_ns()
                _outputs = predict(predictor, _input)
                end = time.time_ns()
                logQueue.put(json.dumps({
                    'action': 'apredict',
                    'expr': 'copy',
                    'start': start - _start,
                    'end': end - _start,
                    'time': end - start
                }))


                start = time.time_ns()
                _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
                end = time.time_ns()
                logQueue.put(json.dumps({
                    'action': 'bparse',
                    'expr': 'copy',
                    'start': start - _start,
                    'end': end - _start,
                    'time': end - start
                }))

            start = time.time_ns()
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': 'ccleanup',
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
            # data = np.ndarray((3840, 2160, 3), dtype=np.uint8, buffer=shm.buf)
            data = torch.frombuffer(shm.buf, dtype=torch.uint8).reshape((2160, 3840, 3))
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
            # image_regions = regionize_image(frame_masked)
            image_regions = [(data, (0, 0))]
            end = time.time_ns()
            logQueue.put(json.dumps({
                'action': '4region',
                'expr': 'shared_memory',
                'start': start - _start,
                'end': end - _start,
                'time': end - start
            }))

            for _image, _offset in image_regions:
                # _outputs = predictor(_image)
                # start = time.time_ns()
                _input = transform_image(predictor, _image, logQueue, 'shared_memory', _start)
                # end = time.time_ns()
                # logQueue.put(json.dumps({
                #     'action': '5transform',
                #     'expr': 'shared_memory',
                #     'start': start - _start,
                #     'end': end - _start,
                #     'time': end - start
                # }))

                start = time.time_ns()
                _outputs = predict(predictor, _input)
                end = time.time_ns()
                logQueue.put(json.dumps({
                    'action': 'apredict',
                    'expr': 'shared_memory',
                    'start': start - _start,
                    'end': end - _start,
                    'time': end - start
                }))

                start = time.time_ns()
                _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
                end = time.time_ns()
                logQueue.put(json.dumps({
                    'action': 'bparse',
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
                'action': 'ccleanup',
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