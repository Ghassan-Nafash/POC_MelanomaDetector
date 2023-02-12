import numpy as np
import matplotlib.pyplot as plt
import cv2
from abc import ABC, abstractclassmethod
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import os
from preprocessing import Preprocessing


class Segmentation(ABC):
    """
    Includes all common methods for skin lesion segmentation.
    Defines methods which the child classes should implement.
    Child classes use different segmentation algorithms, but use already
    tuned parameters for given task.
    """

    @abstractclassmethod
    def display(image, cont):
        """
        To be implemented by every child class.
        Displays the contour on original image.
        """
        pass

    @abstractclassmethod
    def segment(image):
        """
        To be implemented by every child class.
        Returns contour
        """
        pass
    
    def closing(input_img):        
        kernal = np.ones((5, 5), dtype=np.uint8)
        foreground_remove = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernal, iterations=3)
        # plt.imshow(foreground_remove)
        # plt.show()
        return foreground_remove

    def opening(input_img):
        kernal = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernal, iterations=2)
        # plt.imshow(opened)
        # plt.show()
        return opened


'''

Four ways to apply segmentaion, one of them will be picked 

1- MorphACWE
2- ColorFilter
3- thresholding
4- improved_thresholding

'''

class MorphACWE(Segmentation):
    """
    Class for segmentation with Morphological snakes Active contours without edges method.
    Implements methods segment(...) and display(...) of the Segmentation class
    """
    def segment(image) -> np.array:
        """
        segments single image
        """
        image = rgb2gray(image)
        image = img_as_float(image)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)

        ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                    smoothing=3)
        return ls
    

    def display(image, cont):
        """
        Implements the display method of Segmentation class.
        Displays contour on the original image.
        """
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.contour(cont, [0.5], colors="cyan")
        plt.title("Morphological ACWE segmentation", fontsize=12)
        plt.show()


class ColorFilter(Segmentation):

    def segment(input_img) -> np.array:
        '''
        output of this method does not correspond to segmentation,
        it outputs color labeled image for literature purposes only.
        '''

        hsv_roi = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
        
        mask_1 = cv2.inRange(hsv_roi, np.array([0, 20, 20]), np.array([20, 255, 255]))  # Dark Brown or black
        mask_res1 =  cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_1)
        
        mask_2 = cv2.inRange(hsv_roi, np.array([20, 20, 20]), np.array([40, 255, 255])) # Blue or Gray
        mask_res2 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_2)
        
        res1 = cv2.bitwise_or(mask_res1,mask_res2)
        
        mask_3 = cv2.inRange(hsv_roi, np.array([150, 30, 30]), np.array([180, 255, 255])) # pink
        mask_res3 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_3)
        
        mask_4 = cv2.inRange(hsv_roi, np.array([272,63,54]), np.array([282,75,92])) # purple
        mask_res4 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_4)
        
        res2 = cv2.bitwise_or(mask_res3,mask_res4)
        
        final_res = cv2.bitwise_or(res1,res2)


        return final_res
    
    def display(image, cont):
        plt.imshow(image, cmap='hsv')
        plt.show()
        

class BinaryThresholding(Segmentation):
    """
    Class for segmentation with Color filter method.
    To be implemented.
    """
    def segment(image) -> np.array:

        gray_input = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # usual thresholding
        thr, thresh_img = cv2.threshold(gray_input, 128, 255, cv2.THRESH_BINARY)

        return thresh_img
        
    
    def display(image, cont):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.contour(cont, [0.5], colors="cyan")
        plt.title("Morphological ACWE segmentation", fontsize=12)
        plt.show()


class NormalizedOtsuThresholding(Segmentation):
    """
    Class for segmentation with Color filter method.
    To be implemented.
    """
    def segment(image) -> np.array:

        gray_input = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # normalized Otsu
        norm_img = cv2.normalize(gray_input, None, 0, 255, cv2.NORM_MINMAX)
        
        thr, thresh_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        noise = Segmentation.closing(thresh_img)
        # plt.title("Noise Close")
        # plt.imshow(noise,cmap='gray')
        # plt.show()
        noise_open = Segmentation.opening(noise)
        # opened = opening(noise)
        # plt.imshow(noise_open,cmap='gray')
        # plt.title("Noise Open")
        # plt.show()

        image_result = cv2.adaptiveThreshold(noise_open, gray_input.max(), 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)
        # plt.imshow(thresh_img2,cmap='gray')
        # plt.show()
        return image_result
    
    
    def display(image, cont):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.contour(cont, [0.5], colors="cyan")
        plt.title("Morphological ACWE segmentation", fontsize=12)
        plt.show()

