#!/usr/bin/env python3

from utilities import *
from preprocessing import *
import segmentation
import svm
from postprocessing import Postprocessing 
import time
import pandas as pd
import argparse
from visualization import Visualize
import generate_data
import pathlib


def path(p):
    return pathlib.Path(p)


def main():
    parser = argparse.ArgumentParser(prog='PROG',

    description='''this description
                    was indented weird
                    but that is okay''',

    epilog='''
            likewise for this epilog whose whitespace will
            be cleaned up and whose words will be wrapped
            across a couple lines''')

    parser.add_argument("-i", "--input_dir", metavar="IN_DIR", type=path, required=False,
                        help="use --input-dir as the path of the images dataset")
    
    parser.add_argument("-g", "--generate_data", nargs='*', type=path, required=False,
                        help="please provice Metadata path ex HAM10000_metadata.csv")
    
    parser.add_argument("-s", "--svm", metavar="IN_DIR", type=path, required=False,
                        help="use --input-dir as the path of the images dataset")

    args = parser.parse_args()
    

    if args.input_dir:
        Visualize.display_images(args.input_dir)
    if args.generate_data:
        generate_data.generate_dataset(data_path=args.generate_data[0], metadata_path=args.generate_data[1], output_file_name=str(args.generate_data[2]))
    if args.svm:
        model_prediction = svm.Prediction.run_svm(args.svm)


if __name__ == "__main__":
    main()
    