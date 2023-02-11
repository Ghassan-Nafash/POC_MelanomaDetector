'''
This file 

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2


class Preprocessing():
    '''
    different methods of image preprocessing
    '''
    def gamma_correction(input_img, gamma):
        # rase every pixel to the power of gamma ,the larger gamma the darker the image is
        input_img = np.array(input_img/255.0)
        result = np.power(input_img, 1/gamma) * 255
        result = np.array(result).astype(np.uint8)
        
        return result

    # xD  it cannot work, but we can talk about it in the documentation
    def Histogram(input_img):
        # Contrast limited Adaptive Histogramm Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
        clahe_img = clahe.apply(input_img)
        plt.imshow(clahe_img, cmap='gray')
        plt.title("Contrast limited Adaptive Histogram Equalization")
        plt.show()
        return clahe_img
    
    def blur(input_img):
        '''
        Params:
        10 Diameter of the neighborhood, sigma color 50, sigma space 50
        ''' 
        # it preserves hi intensity, it includes also a Gaussian part, also to preserve high edge intensity
        blured = cv2.bilateralFilter(input_img, 25, 60, 60)
        # blured = cv2.GaussianBlur(input_img,(5,5),sigmaX=10,sigmaY=10)
        # blured = cv2.medianBlur(input_img,7)
        return blured
    
    '''
    TODO: integrat closing and opening in one function
    '''

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
    
    
