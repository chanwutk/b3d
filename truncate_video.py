import cv2


LIMIT = 300


cap = cv2.VideoCapture('hwy00.truncated.mp4')
idx = 0

resolution = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter('hwy00.truncated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)

while cap.isOpened() and idx < LIMIT:
    print(idx)
    ret, frame = cap.read()
    if not ret:
        break
    writer.write(frame)
    idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()


fp = open('./hwy00.mp4.truncated.sorted.tracks.jsonl', 'w')
with open('./hwy00.mp4.sorted.tracks.jsonl', 'r') as f:
    for idx, line in enumerate(f):
        if idx >= LIMIT:
            break
        fp.write(line)
fp.close()