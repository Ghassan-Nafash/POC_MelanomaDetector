#!/usr/bin/env python3

from utilities import Utilities
from preprocessing import Preprocessing
from postprocessing import Postprocessing 
import time
from segmentation import *


class ProcessingAlgorithm():
    '''
    using our implementation of other classes 
    for generating features and generating dataset in a CSV format 
    '''

    def process_image(image, image_number):
        '''
        Core of the Algorithm.
        Different steps to process the image        
        '''
        # preprocessing
        gamma_image = Preprocessing.gamma_correction(image, gamma=0.85)

        blured_img = Preprocessing.blur(gamma_image) 

        # segmentation 
        binary_image = NormalizedOtsuWithAdaptiveThresholding.segment(blured_img)
        
        # feature extraction 
        longest_contour = Postprocessing.find_contours(binary_image)     

        features , independent_features = Postprocessing.feature_extractrion(image_number, longest_cntr=longest_contour, image_shape=binary_image.shape)


        return features, independent_features



    def generate_dataset(data_path: str, metadata_path: str, output_file_name):
        '''
        generating features and saving them in CSV file as a table of numbers
        '''
        data_set_path = data_path
        
        data_set = []

        # start timer
        start_time = time.process_time()
        
        # Metadata loading
        # load HAM 10 000 dataset labels        
        dataset_metadata_path = metadata_path
        
        # which labels from metadata we consider malign=positive=1 (others benign=0=negative)
        list_of_malign_labels = ['mel'] 

        meta_data = Utilities.extract_labels_HAM10000(dataset_metadata_path, list_of_malign_labels)
        
        # gen_file_names rename to generate_file_paths
        
        images_paths = Utilities.gen_file_names(data_set_path)
        
        img_count = 0
        img_failed = 0
        for img_path in images_paths:
            # if img_count >= 100: break # for testing only, not using complete dataset
            img_count += 1
            if img_count%10==0: print("Image count: %d / 10000" % img_count)
            img_number = Utilities.extract_img_number(img_path)
            # loading image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            features , independent_features = ProcessingAlgorithm.process_image(img, img_number)                        
            
            img_feature_list = { 'img_number': img_number,
                                
                                'metadata_label': meta_data[img_number], # 1 for malign etc. positive, 0 for benign etc. negative
                                'f_a_0':features[0][0],    
                                'f_a_1':features[0][1], 
                                'f_a_2':features[0][2], 
                                'f_a_3':features[0][3],
                                'f_b_0':features[1],
                                'f_c_0':features[2][0],
                                'f_c_1':features[2][1],
                                'f_c_2':features[2][2],
                                'f_c_3':features[2][3],
                                'f_c_4':features[2][4],
                                
                                'ind_0':independent_features[0],
                                'ind_1':independent_features[1],
                                'ind_2':independent_features[2],
                                'ind_3':independent_features[3],
                                'ind_4':independent_features[4]
                                                            
                                }
            
            if (None in img_feature_list.values()): img_failed += 1
            data_set.append(img_feature_list)
            Utilities.save_dataset(dataset=data_set, file_path='{}{}'.format('./', output_file_name), only_succesfull=True)
            end_time = time.process_time()
            total_time = (end_time - start_time)*1000 # in millis
            avg_time = total_time / img_count
            
        print("total_time = %.0f min" % (total_time/1000/60))
        print("avg_time = %.0f ms per image" % avg_time)
        print("img_failed: %d ... %.1f%% of total images" %(img_failed, img_failed/img_count*100))
        print("img_count", img_count)

        return output_file_name

