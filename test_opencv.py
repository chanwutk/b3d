import time

start = time.time()
import cv2
import torch


filename = 'jnc00.mp4'

cap = cv2.VideoCapture(filename)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('init time:', time.time() - start)

start = time.time()
for i in range(3000):
    ret, frame = cap.read()
    if not ret:
        break
    in_frame = (
        torch
        .from_numpy(frame)
    )
    in_frame.to('cuda:0')

print('FPS:', 3000 / (time.time() - start))