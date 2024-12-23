import argparse
import json

import numpy as np
import shapely
from xml.etree import ElementTree


def get_road(mask) -> shapely.Polygon:
    domains = [
        d.attrib['points'].replace(';', ',')
        for d in mask.findall(f'.//polygon[@label="domain"]')
    ]
    domains = [
        np.array([float(pt) for pt in d.split(',')]).reshape((-1, 2)).tolist()
        for d in domains
    ]
    domains = [shapely.Polygon(d + [d[0]]) for d in domains]
    assert len(domains) == 1, domains
    return domains[0]


def main(args: argparse.Namespace):
    input_file = args.input
    mask = args.mask
    output_file = args.output or input_file + "." + str(mask) + '.jsonl'

    tree = ElementTree.parse(args.mask)
    mask = tree.getroot()
    mask = get_road(mask)

    with open(input_file) as fi:
        with open(output_file, 'w') as fo:
            for line in fi:
                res = json.loads(line)
                detections = res['detections']
                out_detections = []
                for det in detections:
                    bbox = det[:4]
                    x = (bbox[0] + bbox[2]) / 2
                    y = (bbox[1] + bbox[3]) / 2
                    if mask.contains(shapely.geometry.Point(x, y)):
                        out_detections.append(det)
                res['detections'] = out_detections

                fo.write(json.dumps(res) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('-m', '--mask', type=str, help='mask')
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())