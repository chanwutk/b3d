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
import os
from utils import mask_frame, parse_outputs, regionize_image
from xml.etree import ElementTree
import shutil
from matplotlib.path import Path

import queue

from torchvision.transforms import Resize

import torch.nn.functional as F
import PIL.Image as Image

import torch
import numpy.typing as npt


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


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


# def transform_image(model: "DefaultPredictor", original_image):
#     with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
#         # Apply pre-processing to image.
#         if model.input_format == "RGB":
#             # whether the model expects BGR inputs or RGB
#             original_image = original_image[:, :, ::-1]
#         height, width = original_image.shape[:2]
#         image = model.aug.get_transform(original_image).apply_image(original_image)
#         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#         image.to(model.cfg.MODEL.DEVICE)

#         return {"image": image, "height": height, "width": width}


def bin(original_image, model, previous_bin=None):
    if previous_bin is None:
        pass
    else:
        pass


cap = cv2.VideoCapture('jnc00.mp4')
success, frame = cap.read()
print(frame.shape)
cv2.imwrite('frame.jpg', frame)


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
cfg.MODEL.DEVICE = 'cuda:1'
predictor = DefaultPredictor(cfg)

LIMIT = 512

cap = cv2.VideoCapture('jnc00.mp4')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# if os.path.exists('./small-frames'):
#     shutil.rmtree('./small-frames')
# os.mkdir('./small-frames')

tree = ElementTree.parse('jnc00.mp4.mask.xml')
mask = tree.getroot()


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


# bitmap, tl, br = get_bitmap(width, height, mask)
bitmask = get_bitmap(width, height, mask)[0]
bitmask = torch.from_numpy(bitmask).to('cuda:1').to(torch.bool)

# for idx in range(LIMIT):
#     success, frame = cap.read()
#     assert success

#     # mask_frame(frame, mask)
#     frame = torch.from_numpy(frame).to('cuda:1') * bitmask

#     image = transform_image(predictor, frame)
#     print(image['image'].shape)

#     cv2.imwrite(f'./small-frames/{idx:03d}.jpg', image['image'].numpy().transpose((1, 2, 0)))


SIZE = 756, 1344
CHUNK = 32
CHUNK_SIZE = SIZE[1] // CHUNK


def bitmap_index(image_index: int) -> int:
    return int((image_index * SIZE[1]) / CHUNK)


def image_index(bitmap_index: int) -> int:
    return int((bitmap_index * CHUNK) / SIZE[1])


def split_image(image):
    assert image.shape == (*SIZE, 3), (image.shape, SIZE)
    chunks: list[tuple[int, int, npt.NDArray]] = []
    for i in range(round(750 * CHUNK / SIZE[1])):
        for j in range(CHUNK):
            yfrom, yto = bitmap_index(i), bitmap_index(i + 1)
            xfrom, xto = bitmap_index(j), bitmap_index(j + 1)
            chunks.append((i, j, image[yfrom:yto, xfrom:xto, :]))
    return chunks


def get_bbox(detections):
    instances = detections['instances'].to('cpu')
    bboxes = []
    for bbox in instances.pred_boxes:
        bboxes.append(bbox.detach().numpy())
    return bboxes


def mark_detections(detections):
    instances = detections['instances'].to('cpu')
    bboxes = []
    for bbox in instances.pred_boxes:
        bboxes.append(bbox.detach().numpy())
    bitmap = np.zeros((18, 32), dtype=np.int32)

    for bbox in bboxes:
        xfrom, xto = image_index(bbox[0]), image_index(bbox[2])
        yfrom, yto = image_index(bbox[1]), image_index(bbox[3])

        bitmap[yfrom:yto+1, xfrom:xto+1] = 1
    
    return bitmap


def bfs(bitmap: npt.NDArray, i: int, j: int):
    visited: set[tuple[int, int]] = set()
    q = queue.Queue()
    q.put((i, j))
    while not q.empty():
        i, j = q.get()
        if (i, j) in visited:
            continue

        visited.add((i, j))
        for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
            _i += i
            _j += j
            if bitmap[_i, _j] != 0 and (_i, _j) not in visited:
                q.put((_i, _j))
    return list(visited)


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


def pack(bins: list[tuple[int, npt.NDArray, tuple[int, int]]], h: int, w: int):
    bitmap = np.zeros((h, w), dtype=np.bool)

    # first bin to the top left
    _, mask, _ = bins[0]
    bitmap[:mask.shape[0], :mask.shape[1]] = mask

    positions: list[tuple[int, int]] = [(0, 0)]
    for i, (group, mask, _) in enumerate(bins[1:]):
        for i in range(h - mask.shape[0] + 1):
            for j in range(w - mask.shape[1] + 1):
                if not np.any(bitmap[i:i+mask.shape[0], j:j+mask.shape[1]] & mask):
                    bitmap[i:i+mask.shape[0], j:j+mask.shape[1]] |= mask
                    positions.append((i, j))
                    break
            else:
                continue
            break
        else:
            raise ValueError('No space left')
    return bitmap, positions


fourcc = cv2.VideoWriter.fourcc(*'mp4v')

resize = Resize(size=SIZE).to('cuda:1')
# resize = torch.compile(resize)

_idx = 0
# while cap.isOpened():

if os.path.exists('example-frames'):
    shutil.rmtree('example-frames')
os.makedirs('example-frames', exist_ok=True)

LIMIT = 512
bitmaps_size = []
neighbor_size = []
for idx in range(LIMIT):
    _idx += 1
    print(_idx)

    success, frame = cap.read()
    if not success:
        break

    # mask_frame(frame, mask)
    frame = torch.from_numpy(frame).to('cuda:1') * bitmask

    frame = resize(frame.permute(2, 0, 1)).permute(1, 2, 0)

    input = transform_image(predictor, frame)




    # prediction = predictor.model([{'image': torch.from_numpy(frame.astype("float32").transpose(2, 0, 1)).to('cuda:1'), 'height': 750, 'width': 1333}])[0]
    prediction = predictor.model([{'image': input['image'], 'height': SIZE[0], 'width': SIZE[1]}])[0]
    # print(len(prediction['instances'].to('cpu')))

    frame = frame.detach().cpu().numpy()#.transpose((1, 2, 0))
    frame = frame.astype(np.uint8)
    chunks = split_image(frame)

    bitmap = mark_detections(prediction)
    bitmaps_size.append(int(bitmap.sum()))
    neighbor = np.zeros_like(bitmap)

    for i in range(bitmap.shape[0]):
        for j in range(bitmap.shape[1]):
            if bitmap[i, j] == 1:
                neighbor[i, j] = 1
                for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
                    _i += i
                    _j += j
                    if 0 <= _i < bitmap.shape[0] and 0 <= _j < bitmap.shape[1]:
                        neighbor[_i, _j] = 1
    
    if idx % 16 == 0:
        neighbor_size.append((int(bitmap.shape[0] * bitmap.shape[1]), int(bitmap.shape[0] * bitmap.shape[1])))
    else:
        neighbor_size.append((int(neighbor.sum()) + 25 + 16 + 16, int(bitmap.shape[0] * bitmap.shape[1])))

A = 0
B = 0
for a, b in neighbor_size:
    A += a
    B += b
print(A, B)
print((1 - (A / B)) * 100.)
    
with open('neighbor_size.json', 'w') as f:
    json.dump(neighbor_size, f)




















# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Example detection and tracking script')
#     parser.add_argument('-v', '--video', required=True,
#                         help='Input video')
#     parser.add_argument('-c', '--config', required=True,
#                         help='Detection model configuration')
#     parser.add_argument('-m', '--mask', required=True,
#                         help='Mask for the video')
#     return parser.parse_args()


# def main(args):
#     with open(args.config) as fp:
#         config = json.load(fp)
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
#     predictor = DefaultPredictor(cfg)
#     tree = ElementTree.parse(args.mask)
#     mask = tree.getroot()

#     tracker = Sort(max_age=5)
#     cap = cv2.VideoCapture(os.path.expanduser(args.video))
#     trajectories = {}
#     rendering = {}
#     frame_index = 0
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     while cap.isOpened():
#         print('Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
#         success, frame = cap.read()
#         if not success:
#             break
#         frame_masked = mask_frame(frame, mask)

#         image_regions = regionize_image(frame_masked)
#         bboxes = []
#         scores = []
#         for _image, _offset in image_regions:
#             _outputs = predictor(_image)
#             _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
#             bboxes += _bboxes
#             scores += _scores
#         nms_threshold = config['nms_threshold']
#         nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
#         detections = np.zeros((len(nms_bboxes), 5))
#         detections[:, 0:4] = nms_bboxes
#         detections[:, 4] = nms_scores

#         tracked_objects = tracker.update(detections)
#         rendering[frame_index] = []
#         for tracked_object in tracked_objects:
#             tl = (int(tracked_object[0]), int(tracked_object[1]))
#             br = (int(tracked_object[2]), int(tracked_object[3]))
#             object_index = int(tracked_object[4])
#             if object_index not in trajectories:
#                 trajectories[object_index] = []
#             trajectories[object_index].append([
#                 frame_index, tl[0], tl[1], br[0], br[1]])
#             rendering[frame_index].append([
#                 object_index, tl[0], tl[1], br[0], br[1]])

#         frame_index = frame_index + 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

#     scenario = args.video.replace('videos/', '').replace('.mp4', '')
#     with open('output/{}_t.json'.format(scenario), 'w') as fp:
#         json.dump(trajectories, fp)
#     with open('output/{}_r.json'.format(scenario), 'w') as fp:
#         json.dump(rendering, fp)


# if __name__ == '__main__':
#     main(parse_args())
