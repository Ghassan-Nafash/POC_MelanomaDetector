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
    parser = argparse.ArgumentParser(description="call function")

    parser.add_argument("dir", type=path, help="input directories", required=True)

    #parser.add_argument("dir", type=path, help="input images")

    args = parser.parse_args()
    print("args.dir=", args.dir)

    Visualize.display_images(args.dir)

    #if not args.dir:
    #    args.output_dir.mkdir()
    #else:        
        
    #    imageProcessing.display_images(args.dir)


    #print(vars(args))



if __name__ == "__main__":
    data_set_path = "D:/Uni/WS 22-23/Digitale Bildverarbeitung/common_dataset/test"
    Visualize.display_images(data_set_path)
    #test = main()
    #print("test=", test)
    # for documentation reasons
    #compare_segmentation_methods()

    #     
    # metadata path
    #metadata_path = "C:/Users/Yazan/Desktop/DBV literature/Data/HAM10000_metadata.csv"

    #parser.add_argument('--display', choices=imageProcessing.display_images(data_set_path))

    # dataset path
    '''

    # generated data set path
    generated_featuers = 'data_set_v2.csv'


    # config 1 
    imageProcessing.display_images(data_set_path)

    #config 2 : data generation
    generated_data = generate_data.generate_dataset(data_path=data_set_path, metadata_path=metadata_path, generated_file_name="test_data")

    # config 3: run SVM 
    model_prediction = svm.Prediction.run_svm(generated_data)

    '''

    
    