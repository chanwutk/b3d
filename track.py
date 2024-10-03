import json

import numpy as np

from external.sort import Sort
from external.utils import jsonl_reader


def main():
    tracker = Sort(max_age=5)
    rendering = {}
    frame_index = 0
    reader = jsonl_reader('./sorted_detections.jsonl')
    fp = open('tracks.jsonl', 'w')
    for fid, ann in reader:
        print(fid)
        assert fid == frame_index, (fid, frame_index)
        detections = np.array(ann)
        tracked_objects = tracker.update(detections)
        rendering[frame_index] = []
        print("           ", len(tracked_objects) / len(ann), "       ", len(tracked_objects), "/", len(ann))
        for tracked_object in tracked_objects:
            tl = (int(tracked_object[0]), int(tracked_object[1]))
            br = (int(tracked_object[2]), int(tracked_object[3]))
            object_index = int(tracked_object[4])
            rendering[frame_index].append([
                object_index, tl[0], tl[1], br[0], br[1]])
        
        fp.write(json.dumps([frame_index, rendering[frame_index]]) + '\n')
        frame_index = frame_index + 1

    fp.close()


if __name__ == '__main__':
    main()
