#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2


class Preprocessing():
    '''
    different methods for image preprocessing
    '''
    def gamma_correction(input_img, gamma):
        '''
        apply gamma correction on original image
        '''
        # rase every pixel to the power of gamma ,the larger gamma the darker the image is
        input_img = np.array(input_img/255.0)
        result = np.power(input_img, 1/gamma) * 255
        result = np.array(result).astype(np.uint8)
        
        return result

    def Histogram(input_img):
        '''
        Histogram equalization
        were used only for testing purposes
        '''
        # Contrast limited Adaptive Histogramm Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
        clahe_img = clahe.apply(input_img)    

        return clahe_img
    
    def blur(input_img):        
        '''
        blurring input image using bilateral filter
        Params:
        10 Diameter of the neighborhood, sigma color 50, sigma space 50
        ''' 
        # it preserves hi intensity, it includes also a Gaussian part, also to preserve high edge intensity
        blured = cv2.bilateralFilter(input_img, 25, 60, 60)
        # blured = cv2.GaussianBlur(input_img,(5,5),sigmaX=10,sigmaY=10)
        # blured = cv2.medianBlur(input_img,7)
        return blured
    
    
        

