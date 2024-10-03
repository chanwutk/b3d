import cv2
import json

def get_reader(filename):
    annotations_reader = open(filename, 'r')
    cache = {}
    idx = 0
    while True:
        if idx in cache:
            yield cache[idx]
            del cache[idx]
            idx += 1
        annotation_txt = annotations_reader.readline()
        try:
            annotations = json.loads(annotation_txt)
        except json.JSONDecodeError:
            break

        if annotations[0] == idx:
            yield annotations, annotation_txt
            idx += 1
        else:
            cache[annotations[0]] = annotations, annotation_txt
    annotations_reader.close()


reader = get_reader('./tracks.jsonl')

fp = open('./reduced_tracks_2.jsonl', 'w')

for (idx, ann), ann_txt in reader:
    if idx % 10 == 0:
        fp.write(ann_txt)
    else:
        fp.write(json.dumps([idx, [a for a in ann if a[1] < 384 or 3456 < a[1]]]) + '\n')
fp.close()