from utilities import *
from preprocessing import *
import segmentation
from postprocessing import Postprocessing 


if __name__ == "__main__":
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
