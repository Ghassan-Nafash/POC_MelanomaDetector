'''
the Utility file includes the Utility class, 
implementing all the necessary methods for manipulation the images
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

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
                i = int(filename[-6:-1])
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
