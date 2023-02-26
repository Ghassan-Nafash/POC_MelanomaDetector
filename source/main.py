#!/usr/bin/env python3

from utilities import *
from preprocessing import *
import svm
import argparse
from visualization import Visualize
import pathlib
from algorithm import ProcessingAlgorithm
import gui

def path(p):
    '''
    check and return provided path 
    '''
    return pathlib.Path(p)


def main():
    '''
    parse command line(CL) arguments for different system configuration
    '''

    parser = argparse.ArgumentParser(description="This parser is responsible to accept three configurations, \n \
                                    first configuration: display images, \n \
                                    second configuration:generate dataset.csv \n \
                                    third configuration: run Support vector machine for classification")

    parser.add_argument("-i", "--input_dir", metavar="IN_DIR", type=path, required=False,
                        help="use --input_dir as the path of the images dataset")
    
    parser.add_argument("-g", "--generate_data", nargs='*', type=path, required=False,
                        help="3 arguments are needed to run this configurations please provide \n \
                        first: dataset path images, \n \
                        second: metadata path ex HAM10000_metadata.csv, \n \
                        output file name ex output.csv")
    
    parser.add_argument("-s", "--svm", metavar="IN_DIR", type=path, required=False,
                        help="use --svm as the path of the generated dataset.csv to run the SVM")

    parser.add_argument("-p", "--predict", nargs='*', required=False,
                        help="3 arguments are needed to run this configurations please provide \n \
                        first: dataset path images, \n \
                        second: metadata path (for checking correctness) \n \
                        third: image number"
                        )

    parser.add_argument("-G", "--gui", nargs='*', help="use --gui to start the GUI", required=False)
    
    args = parser.parse_args()


    if not (args.input_dir or args.generate_data or args.svm or args.predict or args.gui):
        parser.print_help()
    else:
        if args.input_dir:
            Visualize.display_images(args.input_dir)
            
        if args.generate_data:
            ProcessingAlgorithm.generate_dataset(data_path=args.generate_data[0], 
                                                 metadata_path=args.generate_data[1], output_file_name=str(args.generate_data[2]))
        if args.svm:            
            svm.Prediction.run_svm(args.svm)

        if args.predict:
            svm.Prediction.predict(args.predict[0], args.predict[1], int(args.predict[2]))
        
        if args.gui:
            gui()


if __name__ == "__main__":
    main()
    