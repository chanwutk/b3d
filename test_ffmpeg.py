import time
start = time.time()
import ffmpeg
import torch


filename = 'jnc00.mp4'


probe = ffmpeg.probe(filename)
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
assert video_stream is not None, 'No video stream found'
width = int(video_stream['width'])
height = int(video_stream['height'])


process1 = (
    ffmpeg
    .input(filename)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)
print('init time:', time.time() - start)

start = time.time()
for i in range(3000):
    in_bytes = process1.stdout.read(width * height * 3)
    in_frame = (
        torch
        .frombuffer(in_bytes, dtype=torch.uint8)
        .reshape([height, width, 3])
    )
    in_frame.to('cuda:0')

print('FPS:', 3000 / (time.time() - start))


process1.wait()