import argparse
import json
import os
import shutil
import multiprocessing as mp
import queue

import cv2

from xml.etree import ElementTree

import PIL.Image as Image

from matplotlib.path import Path

import torch
import torch.nn.functional as F
import torch.optim

from torchvision.transforms import Resize

import numpy as np
import numpy.typing as npt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

COLORS_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*COLORS, = map(hex_to_rgb, COLORS_)


SIZE = 756, 1344
CHUNK = 32
CHUNK_SIZE = SIZE[1] // CHUNK


def parse_args():
    parser = argparse.ArgumentParser(description='Train image cell classifier')
    parser.add_argument('-v', '--video',
                        required=False,
                        help='Input video',
                        default='jnc00.mp4')
    parser.add_argument('-g', '--gpu',
                        required=True,
                        help='GPU to use')
    return parser.parse_args()


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
        Image.Resampling.NEAREST: "nearest",
        Image.Resampling.BILINEAR: "bilinear",
        Image.Resampling.BICUBIC: "bicubic",
    }
    mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
    align_corners = None if mode == "nearest" else False
    img = F.interpolate(
        img, (transform.new_h, transform.new_w), mode=mode, align_corners=align_corners
    )
    shape[:2] = (transform.new_h, transform.new_w)
    ret = img.permute(2, 3, 0, 1).view(shape).cpu().numpy()  # nchw -> hw(c)
    
    return ret


def transform_image(model: "DefaultPredictor", original_image):
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


def get_bitmap(width: int, height: int, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    # width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))

    # bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br


def bitmap_index(image_index: int) -> int:
    return int((image_index * SIZE[1]) / CHUNK)


def image_index(bitmap_index: int) -> int:
    return int((bitmap_index * CHUNK) / SIZE[1])


def split_image(image):
    assert image.shape == (*SIZE, 3), (image.shape, SIZE)
    assert image.shape[0] % CHUNK_SIZE == 0
    assert image.shape[1] % CHUNK_SIZE == 0
    chunks = np.ndarray((image.shape[0] // CHUNK_SIZE, image.shape[1] // CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, 3), dtype=image.dtype)
    for i in range(image.shape[0] // CHUNK_SIZE):
        for j in range(image.shape[1] // CHUNK_SIZE):
            yfrom, yto = bitmap_index(i), bitmap_index(i + 1)
            xfrom, xto = bitmap_index(j), bitmap_index(j + 1)
            chunks[i, j] = image[yfrom:yto, xfrom:xto, :]
    return chunks


def get_bbox(detections):
    instances = detections['instances'].to('cpu')
    bboxes: list[npt.NDArray] = []
    for bbox in instances.pred_boxes:
        bboxes.append(bbox.detach().numpy())
    return bboxes


def mark_detections(bboxes: list[npt.NDArray]) -> npt.NDArray:
    bitmap = np.zeros((18, 32), dtype=np.int32)

    for bbox in bboxes:
        xfrom, xto = image_index(bbox[0]), image_index(bbox[2])
        yfrom, yto = image_index(bbox[1]), image_index(bbox[3])

        bitmap[yfrom:yto+1, xfrom:xto+1] = 1
    
    return bitmap


def fill_bitmap(bitmap: npt.NDArray, i: int, j: int):
    value = bitmap[i, j]
    bitmap[i, j] = -1
    q = queue.Queue()
    q.put((i, j))
    filled: list[tuple[int, int]] = []
    while not q.empty():
        i, j = q.get()
        bitmap[i, j] = value
        filled.append((i, j))
        for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
            _i += i
            _j += j
            if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                q.put((_i, _j))
    return filled


def group_chunks(bitmap: npt.NDArray):
    h, w = bitmap.shape
    _groups = np.arange(h * w, dtype=np.int32) + 1
    _groups = _groups.reshape(bitmap.shape)
    _groups = _groups * bitmap

    groups = np.zeros((h + 2, w + 2), dtype=np.int32)
    groups[1:h+1, 1:w+1] = _groups

    visited: set[int] = set()
    bins: list[tuple[int, npt.NDArray, tuple[int, int]]] = []
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            if groups[i, j] == 0 or groups[i, j] in visited:
                continue

            filled = fill_bitmap(groups, i, j)
            filled = np.array(filled, dtype=int).T

            mask = np.zeros((h + 1, w + 1), dtype=np.bool)
            mask[*filled] = True

            offset = np.min(filled, axis=1)
            assert offset.shape == (2,)

            end = np.max(filled, axis=1) + 1
            assert end.shape == (2,)

            mask = mask[offset[0]:end[0], offset[1]:end[1]]
            bins.append((groups[i, j], mask, tuple(offset - 1)))

            visited.add(groups[i, j])
    
    return groups, sorted(bins, key=lambda x: x[1].sum(), reverse=True)


def save_video(videofile: str, gpu: int, inQueue: mp.Queue):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(f'{videofile}.{gpu}.mp4', fourcc, 30, SIZE[::-1])

    while True:
        entry = inQueue.get()
        if entry is None:
            break
        frame, bitmap, bboxes = entry

        marked_frame = frame[:]

        grouped_bitmap, grouped_chunks = group_chunks(bitmap)

        # todo: highlight each cell group with a different color
        for i in range(bitmap.shape[0]):
            for j in range(bitmap.shape[1]):
                if bitmap[i, j] == 1:
                    for idx, (gid, mask, offset) in enumerate(grouped_chunks):
                        _i = i - offset[0]
                        _j = j - offset[1]
                        if _i >= 0 and _j >= 0 and _i < mask.shape[0] and _j < mask.shape[1] and mask[_i, _j]:
                            color = COLORS[idx % len(COLORS)]
                            marked_frame[bitmap_index(i):bitmap_index(i + 1), bitmap_index(j):bitmap_index(j + 1), :] //= 2
                            marked_frame[bitmap_index(i):bitmap_index(i + 1), bitmap_index(j):bitmap_index(j + 1), :] += np.array(color, dtype=np.uint8) // 2
                            break
                    else:
                        marked_frame[bitmap_index(i):bitmap_index(i + 1), bitmap_index(j):bitmap_index(j + 1), :] //= 2
        
        for idx, (gid, mask, offset) in enumerate(grouped_chunks):
            color = COLORS[idx % len(COLORS)]
            for i, j in zip(*np.where(mask)):
                marked_frame[bitmap_index(i + offset[0]):bitmap_index(i + offset[0] + 1), bitmap_index(j + offset[1]):bitmap_index(j + offset[1] + 1), :] //= 2
                marked_frame[bitmap_index(i + offset[0]):bitmap_index(i + offset[0] + 1), bitmap_index(j + offset[1]):bitmap_index(j + offset[1] + 1), :] += np.array(color, dtype=np.uint8) // 2

        # Draw bounding boxes
        marked_frame = np.ascontiguousarray(marked_frame)
        for bb in bboxes:
            x1, y1, x2, y2 = bb
            marked_frame = cv2.rectangle(
                img=marked_frame,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=(0, 255, 0),
                thickness=2,
            )
        
        writer.write(marked_frame)

    writer.release()


def save_images(inQueue: mp.Queue):
    while True:
        entry = inQueue.get()
        if entry is None:
            break
        
        frame, bitmap, _idx = entry

        # split frame into 32 x 18 chunks, each with 42x42 pixels
        chunks = split_image(frame)
        for i in range(frame.shape[0] // CHUNK_SIZE):
            for j in range(frame.shape[1] // CHUNK_SIZE):
                if not np.all(chunks[i, j] == 0):
                    cv2.imwrite(f'./small-frame-chunks/{"1-cars" if int(bitmap[i, j]) else "0-nocars"}/{_idx:03d}_{i:03d}_{j:03d}.jpg', chunks[i, j])


def main(args: argparse.Namespace):
    videofile: str = args.video
    gpu: int = int(args.gpu)

    device = torch.device(f'cuda:{gpu}')
    num_gpus = torch.cuda.device_count()

    cap = cv2.VideoCapture(videofile)
    success, frame = cap.read()
    print(frame.shape)
    cv2.imwrite('frame.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()

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
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(videofile)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_round = frame_count + num_gpus - (frame_count % num_gpus)

    assert frame_count_round % num_gpus == 0
    video_length = frame_count_round // num_gpus
    video_start = video_length * gpu
    video_end = video_length * (gpu + 1)

    if gpu == 0:
        if os.path.exists('./small-frame-chunks'):
            shutil.rmtree('./small-frame-chunks')
        os.mkdir('./small-frame-chunks')

        os.mkdir('./small-frame-chunks/0-nocars')
        os.mkdir('./small-frame-chunks/1-cars')

    tree = ElementTree.parse(f'{videofile}.mask.xml')
    mask = tree.getroot()

    bitmask = get_bitmap(width, height, mask)[0]
    bitmask = torch.from_numpy(bitmask).to(device).to(torch.bool)

    resize = Resize(size=SIZE).to(device)

    saveVideoQueue = mp.Queue()
    pSaveVideo = mp.Process(target=save_video, args=(videofile, gpu, saveVideoQueue))
    pSaveVideo.start()

    saveImageQueue = mp.Queue()
    pSaveImage = mp.Process(target=save_images, args=(saveImageQueue,))
    pSaveImage.start()

    for _idx in range(video_start, video_end):
        print(video_start, '/', _idx, '/', video_end)

        success, frame = cap.read()
        if not success:
            break

        # Mask out the frame
        frame = torch.from_numpy(frame).to(device) * bitmask

        # resize frame to 756x1344
        frame = resize(frame.permute(2, 0, 1)).permute(1, 2, 0)
        assert isinstance(frame, torch.Tensor)

        # transform frame to input format
        input = transform_image(predictor, frame)

        # get prediction
        prediction = predictor.model([{'image': input['image'], 'height': SIZE[0], 'width': SIZE[1]}])[0]

        # convert frame to numpy array
        frame = frame.detach().cpu().numpy()
        frame = frame.astype(np.uint8)

        # get bounding boxes and bitmap of detections
        bboxes = get_bbox(prediction)
        bitmap = mark_detections(bboxes)

        # save images and video
        saveImageQueue.put((frame, bitmap, _idx))
        saveVideoQueue.put((frame, bitmap, bboxes))

    saveVideoQueue.put(None)
    saveImageQueue.put(None)

    pSaveVideo.join()
    pSaveVideo.terminate()

    pSaveImage.join()
    pSaveImage.terminate()


if __name__ == '__main__':
    main(parse_args())
