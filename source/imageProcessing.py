#!/usr/bin/env python3

'''

this script is used for displaying images as the last results.
Just for displying purposes, no image edition here is done.

'''

from utilities import *
from preprocessing import *
import segmentation
from postprocessing import Postprocessing 


def display_images(data_set_path):

    used_method = [ 'Original Image',
                    'gamma correction',
                    'blurring',
                    'NormalizedOtsuWithAdaptiveThresholding',
                    'contours'
                    ]                   

    # load dataset
    images = Utilities.load_all(data_set_path)
    
    for img_number in images.keys():

        img = images[img_number]
        
        # copy image
        image_copy = img.copy()

        # preprocessing
        gamma_image = Preprocessing.gamma_correction(img, gamma=0.85)

        blured_img = Preprocessing.blur(gamma_image)        

        #seg_1 = segmentation.MorphACWE.segment(blured_img)
        seg_3 = segmentation.NormalizedOtsuWithAdaptiveThresholding.segment(blured_img)

        contours = Postprocessing.find_contours(seg_3) 
        __ , features = Postprocessing.feature_extractrion(img_number, longest_cntr=contours, image_shape=blured_img.shape)

        print("Features ",features)
        
        cv2.drawContours(image_copy, [contours], -1, (0,255,0), thickness=5)
        
        images_to_display = [img, gamma_image, blured_img, seg_3, image_copy]

        #Utilities.displayMultiple(images_to_display, used_method, original_img=img, image_num=img_number)
        Utilities.display_for_image_processing(images_to_display, used_method, features=features)
        
                   
        #seg = segmentation.MorphACWE.segment(img)
        #Utilities.display(image=img, cont=result, title="test")        


        '''
        seg_test = segmentation.ColorFilter.segment(img)
        segmentation.ColorFilter.display(seg_test, None)
        '''


if __name__ == '__main__':
    
    path = "C:/Users/Yazan/Desktop/New folder/POC_MelanomaDetector-main/POC_MelanomaDetector-main/Dataset"     
    display_images(path)