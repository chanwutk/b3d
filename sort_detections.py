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
            yield (annotations, annotation_txt)
            idx += 1
        else:
            cache[annotations[0]] = (annotations, annotation_txt)
    annotations_reader.close()


reader = get_reader('/data/chanwutk/projects/b3d-opt/hwy00.mp4.detections.jsonl')

frame_index = 0
fp = open('/data/chanwutk/projects/b3d-opt/hwy00.mp4.sorted.detections.jsonl', 'w')

for (fid, ann), ann_txt in reader:
    print(fid)
    assert fid == frame_index, (fid, frame_index)
    fp.write(ann_txt)
    frame_index += 1

fp.close()

print('done')