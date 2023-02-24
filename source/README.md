# Projekt
We have used a large dataset of images, due to size issues we have to link the source of the dataset:



please use this link to download the provided "dataset_and_Metadata.rar" file.


## Installationsanleitung

Please check if pip is installed

1- create new virtual environment:

	python -m venv \path\to\myenv

2- activate the venv

3- install requiered libraries

	pip install -r requirements.txt

4- call the "main.py" with different configurations flags(-i, -g, -s):

	1. arg : -i is --input_dir, path to images dataset, Action: display last result.
		eg. python .\source\main.py -i "path/to/images_dataset"
	2. arg : -g is generate data, needs three paths (images_data_set_path, path_to_meta_data_.csv, name_of_outputfile.csv)
		eg.
		python .\source\main.py -g "path/to/images_dataset" "HAM10000_metadata.csv" "extracted_features.csv"
	3. arg : -s is to run support Vector machine for classification
		eg.
		python .\source\main.py --svm "data_set_v2.csv"	