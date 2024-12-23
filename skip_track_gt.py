
import argparse
import json

import numpy as np
import shapely
from xml.etree import ElementTree


def main(args: argparse.Namespace):
    input_file = args.input
    skip = args.skip
    output_file = args.output or input_file[:-len('.jsonl')] + ".gt." + str(skip) + '.jsonl'

    with open(input_file) as fi:
        with open(output_file, 'w') as fo:
            for line in fi:
                res = json.loads(line)
                if res['frame_idx'] % skip != 0:
                    continue
                fo.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('-s', '--skip', type=int, help='skip')
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())