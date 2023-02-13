'''
the Utility file includes the Utility class, 
implementing all the necessary methods for manipulation the images
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

import os

class Utilities():
    """
    image manipulation
    """

    def load(path: str, range_start: int, range_end: int) -> dict:
        """
        returns loaded images in RGB format as a dictionary 
            - dict keys are image numbers from the dataset (last 5 digits)
        """
        img_index = list(range(range_start,range_end+1))
        images = dict()
        for i in img_index:
            img_path = path + "ISIC_00" + str(i) + ".jpg"
            img = cv2. imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[i] = img

        return images
    

    # to be tested
    def load_images_in_range(path: str, range_start: int, range_end: int):
        img_index = list(range(range_start,range_end+1))
        images = dict()
        for i in img_index:
            img_path = path + "ISIC_00" + str(i) + ".jpg"
            img = cv2. imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[i] = img

        return images

    def gen_file_names(file_path: str) -> list:
        
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
        

    def display(image, cont, title):
        cv2.drawContours(image, [cont], -1, 255, 2)
        plt.imshow(image, cmap='gray')
        plt.axis('off')        
        plt.title(title, fontsize=12)
        plt.show()


    def displayMultiple(input_images: list, used_method_name: list, original_img, image_num):
        columns = 4
        rows = 1        

        fig = plt.figure(figsize=(17, 4))
        axi = used_method_name

        #a_orig = fig.add_subplot(2, 4, 1)
        #a_orig.set_title("original image")
        #a_orig.imshow(original_img)

        for image in range(len(input_images)):
            ax = fig.add_subplot(1, 4, image+1)
            
            ax.set_title(axi[image])

            ax.imshow(input_images[image], cmap='gray')
        
        fig.suptitle("image number:" + str(image_num))

        plt.show()


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
    

    def write_on_file(file: str, data: list):

        with open('example.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join('{} {}'.format(*tup) for tup in data))
                    
        #my_frame = pandas.DataFrame(features,index=my_index.split(),columns=my_label.split())