import argparse
import glob
import json
import os
import multiprocessing as mp
import time
import logging

import numpy as np
from tqdm import tqdm

from external.nms import nms

def read(filename: str, in_queue: mp.Queue):
    with open(filename, 'r') as fi:
        for line in fi:
            res = json.loads(line)
            if res['action'] == 'prediction':
                in_queue.put(res)
    
    in_queue.put(None)


def main(args: argparse.Namespace):
    logger = logging.getLogger(f'nms')
    logger.setLevel(logging.INFO)


    with open('./configs/config_refined.json') as fp:
        config = json.load(fp)
    input_dir: str = args.input
    input_files = [f for f in os.listdir(input_dir)]

    skip = args.skip
    parent_dir = "/".join(input_dir.split('/')[:-1])
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(parent_dir, 'nms.skip.' + str(skip)))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    in_queue = mp.Queue(maxsize=100)
    processes = []
    for input_file in input_files:
        p = mp.Process(target=read, args=(os.path.join(input_dir, input_file), in_queue))
        p.start()
        processes.append(p)
    
    def process_next(_next_idx):
        start = time.time()
        b, s = [], []
        for detection in idx_to_detections[_next_idx]:
            b += detection['bboxes']
            s += detection['scores']
        nms_threshold = config['nms_threshold']
        nms_bboxes, nms_scores = nms(b, s, nms_threshold)
        detections = np.zeros((len(nms_bboxes), 5))
        detections[:, 0:4] = nms_bboxes
        detections[:, 4] = nms_scores
        end = time.time()
        logger.info(json.dumps({
            'action': 'nms', 'runtime': end - start, 'frame_idx': _next_idx,
            'detections': detections.tolist()
        }))

    done_count = 0
    next_idx = 0
    idx_to_detections = {}
    pbar = tqdm()
    while True:
        res = in_queue.get()
        # print(res)
        if res is None:
            done_count += 1
            if done_count == len(input_files):
                break
            continue
            
        assert res['frame_idx'] >= next_idx, (res, next_idx)

        if res['frame_idx'] not in idx_to_detections:
            idx_to_detections[res['frame_idx']] = []
        idx_to_detections[res['frame_idx']].append(res)

        if next_idx in idx_to_detections:
            if next_idx % skip == 0:
                if len(idx_to_detections[next_idx]) == 1:
                    process_next(next_idx)

                    del idx_to_detections[next_idx]
                    pbar.update(1)
                    next_idx += 1
            else:
                if len(idx_to_detections[next_idx]) == 2:
                    process_next(next_idx)

                    del idx_to_detections[next_idx]
                    pbar.update(1)
                    next_idx += 1
        
    for p in processes:
        p.join()
        p.terminate()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input directory')
    parser.add_argument('-s', '--skip', type=int, help='frames to skip', default=2, required=True)
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())