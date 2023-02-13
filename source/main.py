from utilities import *
from preprocessing import *
import segmentation
from postprocessing import Postprocessing 
import time


def compare_segmentation_methods():
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

    # ISIC_0027861
    
if __name__ == "__main__":

    complete_data_set = dict()

    # start timer
    start_time = time.process_time()

    # gen_file_names rename to generate_file_paths
    generate_images_paths = Utilities.gen_file_names("D:/Uni/WS 22-23/Digitale Bildverarbeitung/common_dataset/Dataset/")

    for img_path in generate_images_paths:

        img_number = Utilities.extract_img_number(img_path)

        img = cv2. imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # preprocessing
        gamma_image = Preprocessing.gamma_correction(img, gamma=0.85)

        blured_img = Preprocessing.blur(gamma_image)  

        binary_image = segmentation.NormalizedOtsuWithAdaptiveThresholding.segment(blured_img)

        longest_contour = Postprocessing.find_contours(binary_image)     

        features = Postprocessing.feature_extractrion(img_number, longest_cntr=longest_contour, image_shape=binary_image.shape)

        item_features = {   'feature_vector':["f_a_0","f_a_1","f_a_2","f_a_3",
                                        "f_b_0",
                                        "f_c_0","f_c_1","f_c_2","f_c_3","f_c_4"],
                            'target_vector':[0,0,0,0,0,0,0,0,0,0],
                            'f_a_0':features[0][0],    
                            'f_a_1':features[0][1], 
                            'f_a_2':features[0][2], 
                            'f_a_3':features[0][3],

                            'f_b_0':features[1],

                            'f_c_0':features[2][0],
                            'f_c_1':features[2][1],
                            'f_c_2':features[2][2],
                            'f_c_3':features[2][3],
                            'f_c_4':features[2][4]                            
                            }
        
        complete_data_set[img_number] = item_features


    end_time = time.process_time()

    total_time = (end_time - start_time)*1000 # in millis

    avg_time = total_time / len(generate_images_paths)

    print("total_time=", total_time)
    print("avg_time=", avg_time)

    print("complete_data_set=", complete_data_set)
    