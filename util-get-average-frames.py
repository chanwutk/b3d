import argparse
import os

import cv2
import numpy as np


def main(args):
    input_dir = args.input
    output_dir = args.output
    num_frames = args.num_frames

    for file in os.listdir(input_dir):
        if file.endswith('.mp4'):
            print('Processing file:', file)
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file.replace('.mp4', '.jpg'))

            cap = cv2.VideoCapture(input_file)
            assert cap.isOpened()

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert num_frames <= length

            frame_indices = [i * (length // num_frames) for i in range(num_frames)]
            sum_frames = None
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                success, frame = cap.read()
                assert success

                frame = frame.astype(np.int32)
                if sum_frames is None:
                    sum_frames = np.zeros_like(frame)
                sum_frames += frame

            assert sum_frames is not None
            sum_frames = (sum_frames // len(frame_indices)).astype(np.uint8)
            cv2.imwrite(output_file, sum_frames)

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input video dir', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output frame dir', required=True)
    parser.add_argument('-n', '--num_frames', type=int, help='Number of frames to extract', default=128, required=False)
    main(parser.parse_args())
