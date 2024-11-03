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


def main(args: argparse.Namespace):
    input_file = args.input
    output_file = args.output
    if output_file is None:
        output_file = input_file.replace('nms', 'tracks') + '.jsonl'
    tracker = Sort(max_age=5)
    frame_index = 0
    reader = jsonl_reader(input_file)

    tree = ElementTree.parse(args.mask)
    mask = tree.getroot()
    roads: list[tuple[int, shapely.Polygon]] = []
    for s in [0, 1, 2, 3, 4, 5]:
        domains = get_road(mask, s)
        for d in domains:
            roads.append((s, d))

    trajectories = {}
    with open(input_file) as fi:
        fp = open(output_file, 'w')
        for line in fi:
            res = json.loads(line)
            assert res['frame_idx'] == frame_index, (res['frame_idx'], frame_index)
            ann = res['detections']
            start = time.time()
            detections = np.array(ann)
            tracked_objects = tracker.update(detections)
            rendering = []
            # print("           ", len(tracked_objects) / len(ann), "       ", len(tracked_objects), "/", len(ann))
            min_skip = 10
            skip_time = 0
            for tracked_object in tracked_objects:
                tl = (int(tracked_object[0]), int(tracked_object[1]))
                br = (int(tracked_object[2]), int(tracked_object[3]))

                object_index = int(tracked_object[4])
                rendering.append([
                    object_index, tl[0], tl[1], br[0], br[1]])

                if object_index not in trajectories:
                    trajectories[object_index] = []
                trajectories[object_index].append([
                    frame_index, tl[0], tl[1], br[0], br[1]])

                skip_start = time.time()
                tr = (int(tracked_object[2]), int(tracked_object[1]))
                bl = (int(tracked_object[0]), int(tracked_object[3]))
                box = shapely.Polygon([tl, tr, br, bl, tl])

                max_area = -1
                max_s = None
                for _s, _r in roads:
                    area = _r.intersection(box).area
                    if area > 0 and area > max_area:
                        max_area = area
                        max_s = _s
                
                if max_s is not None:
                    min_skip = min(min_skip, max_s)
                skip_end = time.time()
                skip_time += skip_end - skip_start
            
            if min_skip == 10:
                min_skip = 0
            print(min_skip * 30)
            fp.write(json.dumps({
                'action': 'skip',
                'frame_idx': frame_index,
                'tracks': 1 if min_skip == 0 else min_skip * 30,
                'runtime': skip_time
            }) + '\n')
            
            fp.write(json.dumps({
                'action': 'tracks',
                'frame_idx': frame_index,
                'tracks': rendering,
                'runtime': time.time() - start
            }) + '\n')
            frame_index = frame_index + 1

        fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input detection file', default='hwy00.mp4.sorted.detections.jsonl')
    parser.add_argument('-o', '--output', type=str, help='Output track file', default=None, required=False)
    parser.add_argument('-m', '--mask', type=str, help='Mask file', default=None, required=True)
    main(parser.parse_args())
