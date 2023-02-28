#!/usr/bin/env python3

import matplotlib.pyplot as plt
from postprocessing import Postprocessing 
import pandas as pd
from pandas.plotting import radviz
from utilities import *
from preprocessing import *
import segmentation


class Visualize():
    '''
    visualization for final and subsequent results
    of image processing
    '''

    def plot_features(training_data):
        '''show features dependency'''

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
        ''' displaying subsequent steps of the processing algorithm'''

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

            features, independent_features = Postprocessing.feature_extractrion(img_number, longest_cntr=contours, image_shape=blured_img.shape)            
            
            cv2.drawContours(image_copy, [contours], -1, (0,255,0), thickness=5)
            
            images_to_display = [img, gamma_image, blured_img, seg_3, image_copy]

            #Utilities.displayMultiple(images_to_display, used_method, original_img=img, image_num=img_number)
            Visualize.display_for_image_processing(images_to_display, used_method, features=features, independent_features=independent_features)
            
                    
            #seg = segmentation.MorphACWE.segment(img)
            #Utilities.display(image=img, cont=result, title="test")        


            '''
            seg_test = segmentation.ColorFilter.segment(img)
            segmentation.ColorFilter.display(seg_test, None)
            '''
            
    def compare_segmentation_methods():
        """
        Used for testing and documentation only,
        compairing performance of different segmentation methods
        """
        
        used_method = [ 'BinaryThresholding',
                        'NormalizedOtsuThresholding',
                        'NormalizedOtsuWithAdaptiveThresholding',
                        'MorphACWE'
                        ]
                
        start_index = 29422
        end_index = 29429
        images = Utilities.load_all("path/to/dataset")
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

            Visualize.displayMultiple(images_to_display, used_method, original_img=img, image_num=img_number)
            
            result = Postprocessing.find_contours(seg_1)            
            #seg = segmentation.MorphACWE.segment(img)
            #Utilities.display(image=img, cont=result, title="test")        

            '''
            seg_test = segmentation.ColorFilter.segment(img)
            segmentation.ColorFilter.display(seg_test, None)
            '''


    def display(image, cont, title):
        cv2.drawContours(image, [cont], -1, 255, 2)
        plt.imshow(image, cmap='gray')
        plt.axis('off')        
        plt.title(title, fontsize=12)
        plt.show()


    def display_for_image_processing(input_images: list, used_method_name: list, features, independent_features):
        '''
        display images in a step-wise as they are pre and post processed.
        plotting features 
        First paper: Melanoma Skin Cancer Detection Using Image Processing and Machine Learning Techniques 
        second paper: Computer aided Melanoma skin cancer detection using Image Processing
        '''

        fig = plt.figure(figsize=(15, 5))
        axi = used_method_name
        
        # __ , features = Postprocessing.feature_extractrion(img_number, longest_cntr=longest_contour, image_shape=binary_image.shape)
        
        first_paper_img = np.ones((750,850))

        second_paper_img = np.ones((750,850))

        independent_features_img = np.ones((750,850))

        first_paper_features_names = ['ir_A', 'Circularity', 'circile_irrigularity', 'ir_Abnormality']

        second_paper_features_names = ['ir_A', 'ir_B', 'ir_C', 'ir_D', 'Circularity_index']
        
        independent_feature_title = ["perimeter","largest_area","minor_diameter","major_diameter","ellipse_irrigularity"]

        list_of_images = [first_paper_img, second_paper_img, independent_features_img]
        
        features_list = [features[0], features[2], independent_features]

        feature_names = [first_paper_features_names, second_paper_features_names, independent_feature_title]

        list_of_feature_sets = ['Thaajwer 2020: features', 'Shivangi 2015: features', 'independent features']

        for i in range(3):

            features_in = features_list[i]
            image = list_of_images[i]
            hight_indentation = 0
            title = list_of_feature_sets[i]
            
            for count, feature in enumerate(features_in):
                
                ax = fig.add_subplot(2, 4, 2+i)

                ax.set_axis_off()           
                
                # hanle images with no contours
                if feature != None:

                    cv2.putText(image, feature_names[i][count] + (": %.2f "%feature), (50, 80 + hight_indentation),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=0, thickness=3)
                    
                    ax.imshow(image,cmap='gray')

                    ax.set_title(title)

                    hight_indentation += 100
                        
        # original image
        ax = fig.add_subplot(2, 4, 1)            
        ax.set_title(axi[0])
        ax.imshow(input_images[0])        
        # gamma correction
        ax = fig.add_subplot(2, 4, 5)            
        ax.set_title(axi[1])
        ax.imshow(input_images[1])        
        # blured
        ax = fig.add_subplot(2, 4, 6)            
        ax.set_title(axi[2])
        ax.imshow(input_images[2])        
        # segmentation
        ax = fig.add_subplot(2, 4, 7)            
        ax.set_title(axi[3])
        ax.imshow(input_images[3], cmap='gray')    
        # contour
        ax = fig.add_subplot(2, 4, 8)            
        ax.set_title(axi[4])
        ax.imshow(input_images[4])
        
        plt.tight_layout()
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
            
            ax = fig.add_subplot(2, 4, image+4)
            
            ax.set_title(axi[image])
            
            if image == 3:
                ax.imshow(input_images[image], cmap='gray')
            else:
                ax.imshow(input_images[image])
        
        fig.suptitle("image number:" + str(image_num))

        plt.show()

