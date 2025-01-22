import argparse

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.typing as npt

def resize(image: npt.NDArray, size):
    transform = transforms.Compose([transforms.Resize(size)])
    return transform(torch.from_numpy(image.transpose(2, 0, 1))).numpy().transpose(1, 2, 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Example resize script')
    parser.add_argument('-i', '--input', required=True,
                        help='Input image')
    parser.add_argument('-o', '--output', required=False,
                        help='Output image')
    parser.add_argument('-s', '--size', required=False,
                        help='Output image size as <HEIGHT>x<WIDTH>')
    return parser.parse_args()


def main(args):

    cap = cv2.VideoCapture(args.input)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        print('Error: Could not open video.')
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    size = tuple(map(int, args.size.split('x')))

    output = args.output or ('.'.join(args.input.split('.')[:-1]) + '_resized.mp4')
    writer = cv2.VideoWriter(output, cv2.VideoWriter.fourcc(*'mp4v'), fps, size[::-1])

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or idx > 400:
                break
            idx += 1
            if idx < 25:
                continue

            print(frame.shape)
            frame = resize(frame, size)
            print(frame.shape)
            writer.write(frame)
    except KeyboardInterrupt:
        pass
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main(parse_args())