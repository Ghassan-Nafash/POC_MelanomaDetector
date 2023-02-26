#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from postprocessing import Postprocessing
import os

class Utilities():
    """
    implementing all the necessary methods for manipulation the images
    """    

    def load_images_in_range(path: str, range_start: int, range_end: int):
        '''
        load specific images identified with image number leaving first two digits        
        eg. original image number = ISIC_0024306
        to call function send -> image number = 24306
        '''
        
        img_index = list(range(range_start,range_end+1))
        images = dict()
        for i in img_index:
            img_path = path + "\ISIC_00" + str(i) + ".jpg"
            img = cv2. imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[i] = img

        return images


    def gen_file_names(file_path: str) -> list:
        '''
        generate list of names from single directory path
        and checking a valid .jpg extention
        '''

        images = [] 
        valid_ext = [".jpg"]
        for f in os.listdir(file_path):
            filename = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_ext:
                os.path.join(file_path, f)
                images.append(os.path.join(file_path, f))
        
        return images
    
    
    def extract_img_number(image_name: str):              
        ''' extracting image number '''  
        valid_ext = [".jpg"]
        
        filename = os.path.splitext(image_name)[0]

        ext = os.path.splitext(image_name)[1]

        if ext.lower() in valid_ext:
            image_number = int(filename.split('_')[-1])


        return image_number


    def load_all(path: str):
        """
        returns loaded images in RGB format as a dictionary 
            - dict keys are image numbers from the dataset (last 5 digits)
        """
        images = dict()
        valid_ext = [".jpg"]
        for f in os.listdir(path):
            filename = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_ext:
                i = int(filename.split('_')[1])

                img = cv2. imread(os.path.join(path, f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[i] = img

        return images            


    def save_dataset(dataset: list, file_path: str, only_succesfull=True):
        """
        Saves a dataset (list of dictionaries) in csv format to a specified location on disk
        """
        # Write the dataset to the CSV file
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = [str(i) for i in dataset[0].keys()]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data in dataset:
                if only_succesfull:
                    if not None in data.values():
                        writer.writerow(data)
                elif not only_succesfull:
                    writer.writerow(data)


    def extract_labels_HAM10000(dataset_metadata_path, list_of_malign_labels):
        """
        Exctracts labels (malign=1, benign=0) for individual images from HAM 10 000 dataset. 
        return: dictionary with 5-digit img_numbers as keys and label as values
        """
        meta_data = dict()
        with open(dataset_metadata_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row_data = dict(row)
                image_number = int(row_data['image_id'].split('_')[-1])
                image_class = int(row_data['dx'] in list_of_malign_labels) # 1 for positive, 0 for negative
                meta_data[image_number] = image_class
        return meta_data

    