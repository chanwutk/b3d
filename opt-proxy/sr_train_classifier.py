import argparse
import json
import os
import shutil
import multiprocessing as mp
import queue
import time

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

LIMIT = 4096
SAMPLINGS = 16
SIZE = 756, 1344
CHUNK = 32
CHUNK_SIZE = SIZE[1] // CHUNK


def parse_args():
    parser = argparse.ArgumentParser(description='Train image cell classifier')
    parser.add_argument('-v', '--video',
                        required=False,
                        help='Input video',
                        default='jnc00.mp4')
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


def save_images(inQueue: mp.Queue):
    if os.path.exists('./sr-classifier-train-data'):
        shutil.rmtree('./sr-classifier-train-data')
    os.makedirs('./sr-classifier-train-data/0-nocars')
    os.makedirs('./sr-classifier-train-data/1-cars')
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
                    cv2.imwrite(f'./sr-classifier-train-data/{"1-cars" if int(bitmap[i, j]) else "0-nocars"}/{_idx:03d}_{i:03d}_{j:03d}.jpg', chunks[i, j])


def prepare_data(videofile: str, device: torch.device, predictor: DefaultPredictor, bitmask: torch.Tensor, resize: Resize):
    cap = cv2.VideoCapture(videofile)

    saveImageQueue = mp.Queue()
    pSaveImage = mp.Process(target=save_images, args=(saveImageQueue,))
    pSaveImage.start()

    for idx in tqdm(range(LIMIT)):
        success, frame = cap.read()

        if idx % SAMPLINGS != 0:
            continue

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
        saveImageQueue.put((frame, bitmap, idx))

    saveImageQueue.put(None)

    pSaveImage.join()
    pSaveImage.terminate()


import torch
import torch.utils.data
import torch.optim

from torchvision import datasets, transforms
from torchvision.models.efficientnet import efficientnet_v2_s
from torch.optim import Adam  # type: ignore

from tqdm import tqdm


def custom_efficientnet(device: torch.device):
    model = efficientnet_v2_s(pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=1, bias=True).to(device)

    return model


def train_step(model, loss_fn, optimizer, inputs, labels):
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs, device):
    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in tqdm(train_loader, total=len(train_loader)): #iterate ove batches
            x_batch = x_batch.to(device) #move to gpu
            y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
            y_batch = y_batch.to(device) #move to gpu

            loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)

            epoch_loss += loss / len(train_loader)
            losses.append(loss)
        
        epoch_train_losses.append(epoch_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

        #validation doesnt requires gradient
        with torch.no_grad():
            model.eval()
            cumulative_loss = 0
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
                y_batch = y_batch.to(device)

                #model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat,y_batch)
                cumulative_loss += loss / len(test_loader)
                val_losses.append(val_loss.item())


            epoch_test_losses.append(cumulative_loss)
            print('Epoch : {}, val loss : {}'.format(epoch+1,cumulative_loss))  
            
            best_loss = min(epoch_test_losses)
            
            #save best model
            if cumulative_loss <= best_loss:
                best_model_wts = model.state_dict()
            
            # #early stopping
            # early_stopping_counter = 0
            # if cum_loss > best_loss:
            #   early_stopping_counter +=1

            # if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            #   print("/nTerminating: early stopping")
            #   break #terminate training
    
    return best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses


def train_proxy(device: torch.device):
    model = custom_efficientnet(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()    
    # optimizer = Adam(model.parameters(), lr=0.001)
    running_loss = 0.0


    train_data = datasets.ImageFolder('./sr-classifier-train-data', transform=transforms.ToTensor())

    generator = torch.Generator().manual_seed(42)
    split = int(0.8 * len(train_data))
    train_data, test_data = torch.utils.data.random_split(
        dataset=train_data,
        lengths=[split, len(train_data) - split],
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True)

    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    n_epochs = 10
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03

    print("Training FC")
    optimizer = Adam(model.classifier[-1].parameters(), lr=0.001)
    best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=3, device=device)
    model.load_state_dict(best_model_wts)

    print("Tuning all layers")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = Adam(model.parameters(), lr=0.001)
    best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=20, device=device)
    model.load_state_dict(best_model_wts)

    #load best model
    model.load_state_dict(best_model_wts)

    return model


def main(args: argparse.Namespace):
    videofile: str = args.video
    gpu: int = 0

    device = torch.device(f'cuda:{gpu}')

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
    cap.release()

    tree = ElementTree.parse(f'{videofile}.mask.xml')
    mask = tree.getroot()
    bitmask = get_bitmap(width, height, mask)[0]
    bitmask = torch.from_numpy(bitmask).to(device).to(torch.bool)

    resize = Resize(size=SIZE).to(device)

    start = time.time()
    prepare_data(videofile, device, predictor, bitmask, resize)
    end = time.time()
    print(f'Time taken: {end - start} seconds')

    start = time.time()
    model = train_proxy(device)
    torch.save(model, 'model.pth')
    end = time.time()
    print(f'Time taken: {end - start} seconds')


if __name__ == '__main__':
    main(parse_args())
