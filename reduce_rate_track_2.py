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


for rr in [2, 4, 8, 16]:
    reader = get_reader('/data/chanwutk/projects/b3d-opt/jnc00.mp4.sorted.tracks.jsonl')

    fp = open(f'/data/chanwutk/projects/b3d-opt/jnc00.mp4.sorted.rr{rr}.tracks.jsonl', 'w')

    for (idx, ann), ann_txt in reader:
        if idx % rr == 0:
            fp.write(ann_txt)
        else:
            fp.write(json.dumps([idx, [a for a in ann if a[1] < 384 or 3456 < a[1] or (1536 < a[1] and a[1] < 3072)]]) + '\n')
    fp.close()