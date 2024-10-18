import argparse
import json
import os


def main(args: argparse.Namespace):
    input_file: str = args.input
    keep: list[int] = [int(k) for k in args.keep.split(',')]
    skip = args.skip
    output_file: str = args.output or f'{input_file}.filtered.{skip}.jsonl'

    with open(output_file, 'w') as fo:
        with open(input_file, 'r') as fi:
            for line in fi:
                record = json.loads(line)
                if record['action'] == 'init':
                    fo.write(line)
                elif record["region_idx"] in keep or record['frame_idx'] % skip == 0:
                    fo.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input video file', default='./hwy00.mp4')
    parser.add_argument('-k', '--keep', type=str, help='Comma separated regions to keep', default='', required=True)
    parser.add_argument('-s', '--skip', type=int, help='frames to skip', default=30, required=True)
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())