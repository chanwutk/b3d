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
            yield annotations
            idx += 1
        else:
            cache[annotations[0]] = annotations
    annotations_reader.close()


cap = cv2.VideoCapture('./hwy00.mp4')
frame_index = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
predictor_count = 0
mask_array = None
reader = get_reader('./detections.json')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    fid, ann = next(reader)
    print(fid)
    assert fid == frame_index, (fid, frame_index)

    for bbox in ann:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    out.write(frame)

    frame_index += 1
    # if frame_index > 30:
    #     break
print('done')

out.release()
cap.release()
cv2.destroyAllWindows()