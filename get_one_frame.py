import argparse
import cv2


def main(args):
    input_file = args.input
    output_file = args.output

    if output_file is None:
        output_file = input_file + '.jpg'

    cap = cv2.VideoCapture(input_file)
    assert cap.isOpened()

    success, frame = cap.read()
    assert success

    cv2.imwrite(output_file, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input video file', default='hwy00.mp4')
    parser.add_argument('-o', '--output', type=str, help='Output frame file', default=None, required=False)
    main(parser.parse_args())
