import argparse
import jpype
import jpype.imports
# from jpype.types import *

# Launch the JVM
jpype.startJVM(classpath=['DalsooBinPacking/target/dalsoopack-0.9-SNAPSHOT.jar'])

# import the Java modules
from whitegreen.dalsoo import PackedPoly, Bin

print(dir(PackedPoly))

PackedPoly(1, [[1, 2], [3, 4]])


def main(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('-m', '--mask', type=str, help='mask')
    parser.add_argument('-o', '--output', type=str, help='Output detections file', default=None, required=False)
    main(parser.parse_args())
