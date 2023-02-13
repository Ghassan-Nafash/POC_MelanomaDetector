from utilities import *
from preprocessing import *
import segmentation
from postprocessing import Postprocessing 
import time


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

    # ISIC_0027861
    
if __name__ == "__main__":
    # data_set_path = "D:/Uni/WS 22-23/Digitale Bildverarbeitung/common_dataset/Dataset/" # Ghassan
    data_set_path = "C:/Users/ancik/Documents/GitHub/archive/HAM10000_images/"

    data_set = []

    # start timer
    start_time = time.process_time()

    # Metadata loading
    # load HAM 10 000 dataset labels
    dataset_metadata_path = "C:/Users/ancik/Documents/GitHub/archive/HAM10000_metadata.csv"
    # which labels from metadata we consider malign=positive=1 (others benign=0=negative)
    list_of_malign_labels = ['mel', 'bcc'] # bcc rarely metastizes
    meta_data = Utilities.extract_labels_HAM10000(dataset_metadata_path, list_of_malign_labels)

    # gen_file_names rename to generate_file_paths
    images_paths = Utilities.gen_file_names(data_set_path)

    img_count = 0
    img_failed = 0
    for img_path in images_paths:
        # if img_count >= 100: break # for testing only, not using complete dataset
        img_count += 1
        if img_count%10==0: print("Image count: %d" % img_count)

        img_number = Utilities.extract_img_number(img_path)

        # loading image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # preprocessing
        gamma_image = Preprocessing.gamma_correction(img, gamma=0.85)
        blured_img = Preprocessing.blur(gamma_image)  

        binary_image = segmentation.NormalizedOtsuWithAdaptiveThresholding.segment(blured_img)

        # feature extraction 
        longest_contour = Postprocessing.find_contours(binary_image)     
        features = Postprocessing.feature_extractrion(img_number, longest_cntr=longest_contour, image_shape=binary_image.shape)

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
                            'f_c_4':features[2][4]                            
                            }
        if (None in img_feature_list.values()): img_failed += 1

        data_set.append(img_feature_list)

    Utilities.save_dataset(dataset=data_set, file_path="./data_set.csv", only_succesfull=True)

    end_time = time.process_time()

    total_time = (end_time - start_time)*1000 # in millis

    avg_time = total_time / img_count

    print("total_time = %.0f min" % (total_time/1000/60))
    print("avg_time = %.0f ms per image" % avg_time)
    print("img_failed: %d ... %.1f%% of total images" %(img_failed, img_failed/img_count*100))
    print("img_count", img_count)

    # print("complete_data_set=", data_set)
    