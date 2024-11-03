import argparse
import json
import time
from xml.etree import ElementTree

import numpy as np

from external.sort import Sort
from external.utils import jsonl_reader

import shapely


def get_road(mask, section: int):
    domains = [
        d.attrib['points'].replace(';', ',')
        for d in mask.findall(f'.//polygon[@label="road{section}"]')
    ]
    domains = [
        np.array([float(pt) for pt in d.split(',')]).reshape((-1, 2)).tolist()
        for d in domains
    ]
    domains = [shapely.Polygon(d + [d[0]]) for d in domains]
    return domains


SKIP = True


def main(args: argparse.Namespace):
    input_file = args.input
    output_file = args.output
    if output_file is None:
        output_file = input_file.replace('nms', 'tracks') + f'.runtime{'' if SKIP else '.noskip'}.jsonl'
    tracker = Sort(max_age=5)
    frame_index = 0
    reader = jsonl_reader(input_file)

    trajectories = {}
    frames = set()
    nxt = None
    with open(input_file) as fi:
        for line in fi:
            res = json.loads(line)
            assert res['frame_idx'] == frame_index // 2, (res['frame_idx'], frame_index)
            action = res['action']
            frame_index = frame_index + 1
            if action != 'skip':
                continue

            if nxt is None or nxt == res['frame_idx']:
                nxt = res['frame_idx'] + res['tracks'] + 1
                frames.add(res['frame_idx'])
                print(nxt)
            

    fp = open(output_file, 'w')
    with open(input_file) as fi:
        for line in fi:
            res = json.loads(line)
            action = res['action']

            if SKIP and res['frame_idx'] not in frames:
                continue

            if not SKIP and action == 'skip':
                continue

            fp.write(json.dumps({
                'skip': SKIP,
                'action': action,
                'frame_idx': res['frame_idx'],
                'runtime': res['runtime'],
            }) + ',\n')


    with open('jnc00.mp4.results/nms.skip.1') as fi:
        for line in fi:
            res = json.loads(line)
            action = res['action']

            if SKIP and res['frame_idx'] not in frames:
                continue

            fp.write(json.dumps({
                'skip': SKIP,
                'action': action,
                'frame_idx': res['frame_idx'],
                'runtime': res['runtime'],
            }) + ',\n')
    
    for i in range(6):
        with open(f'jnc00.mp4.results/detection/detection.{i}.log.jsonl') as fi:
            for line in fi:
                res = json.loads(line)
                action = res['action']

                if action == 'init':
                    continue

                if SKIP and res['frame_idx'] not in frames:
                    continue

                fp.write(json.dumps({
                    'skip': SKIP,
                    'action': action,
                    'frame_idx': res['frame_idx'],
                    'runtime': res['runtime'],
                }) + ',\n')
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input detection file', default='hwy00.mp4.sorted.detections.jsonl')
    parser.add_argument('-d', '--detection', type=str, help='Output track file', default=None, required=False)
    parser.add_argument('-o', '--output', type=str, help='Output track file', default=None, required=False)
    main(parser.parse_args())
