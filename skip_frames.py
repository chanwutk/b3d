import argparse
import json
import os


def main(args: argparse.Namespace):
    input_dir: str = args.input
    input_files: list[str] = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]

    keep: list[int] = [int(k) for k in args.keep.split(',')]
    skips: list[int] = [int(s) for s in args.skip.split(',')]

    for skip in skips:
        output_dir: str = args.output or input_dir + '.skip.' + str(skip)

        for input_file in input_files:
            with open(os.path.join(output_dir, input_file), 'w') as fo:
                with open(os.path.join(input_dir, input_file), 'r') as fi:
                    for line in fi:
                        record = json.loads(line)
                        if record['action'] == 'init':
                            fo.write(line)
                        elif record["region_idx"] in keep or record['frame_idx'] % skip == 0:
                            fo.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input detection directory')
    parser.add_argument('-k', '--keep', type=str, help='Comma separated regions to keep', default='', required=True)
    parser.add_argument('-s', '--skip', type=str, help='frames to skip', default='2', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())