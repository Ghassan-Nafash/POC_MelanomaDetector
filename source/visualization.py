#!/usr/bin/env python3

import matplotlib.pyplot as plt
from postprocessing import Postprocessing 
import svm
import pandas as pd
from pandas.plotting import radviz
from utilities import *
from preprocessing import *
import segmentation


class Visualize():

    def plot_features(training_data):
 
        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        X = x_train
        y = y_train
        plt.scatter(X['ind_0'],X['ind_1'], c=y, s=30,cmap='seismic')
        plt.scatter(X['ind_2'],X['ind_3'], c=y, s=30,cmap='seismic')
        plt.scatter(X['ind_4'], c=y, s=30,cmap='seismic')
        plt.show()
    

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
            
    def compare_segmentation_methods():
        """
        Used for testing and documentation only
        """
        used_method = [ 'BinaryThresholding',
                        'NormalizedOtsuThresholding',
                        'NormalizedOtsuWithAdaptiveThresholding',
                        'MorphACWE'
                        ]
                
        start_index = 29422
        end_index = 29429
        images = Utilities.load_all("D:/Uni/WS 22-23/Digitale Bildverarbeitung/common_dataset/Dataset/")
        # images = Segmentation.load_all("C:/Users/ancik/Documents/GitHub/Dataset/")
        for img_number in images.keys():
            img = images[img_number]
            
            plt.imshow(img)
            plt.show()
            # preprocessing
            gamma_image = Preprocessing.gamma_correction(img, gamma=0.85)
            blured_img = Preprocessing.blur(gamma_image)        

            seg_1 = segmentation.BinaryThresholding.segment(blured_img)

            seg_2 = segmentation.NormalizedOtsuThresholding.segment(blured_img)

            seg_3 = segmentation.NormalizedOtsuWithAdaptiveThresholding.segment(blured_img)

            seg_4 = segmentation.MorphACWE.segment(blured_img)
            
            images_to_display = [seg_1, seg_2, seg_3, seg_4]

            Utilities.displayMultiple(images_to_display, used_method, original_img=img, image_num=img_number)
            
            result = Postprocessing.find_contours(seg_1)            
            #seg = segmentation.MorphACWE.segment(img)
            #Utilities.display(image=img, cont=result, title="test")        

            '''
            seg_test = segmentation.ColorFilter.segment(img)
            segmentation.ColorFilter.display(seg_test, None)
            '''

if __name__ == "__main__":
    
    training_data = pd.read_csv('data_set_v2.csv' , index_col=0)
    pd.plotting.scatter_matrix(training_data[['ind_0','ind_1','ind_2','ind_3']], alpha=0.2)
    plt.show()
    
    