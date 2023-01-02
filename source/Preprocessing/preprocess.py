#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import openCVTools as cvt


#resources
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

#TODO: texture synthesis
#implemented inside E:\Uni\WS 22-23\Digitale Bildverarbeitung\texture_repo
#paper is also exist 
#another source 
#https://github.com/spieswl/magic-eraser



def filterImage(img):
    
    # Otsu's thresholding after Gaussian filtering
    kernel = np.ones((5, 5), np.uint8)

    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    image = th3

    # detect edges in photo
    edges = detect_edges(image)

    
    # fatten the long edges
    #element   = cv.getStructuringElement(cv.MORPH_DILATE, (3, 3))
    #edgesLong = cv.dilate(edges, element)


    #img_erosion = cv.erode(edges, kernel, iterations=1)
    #img_dilation = cv.dilate(edges, kernel, iterations=1)
    
    #opening = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    return closing


def detect_edges(filtered_img):
    edges = cv.Canny(filtered_img,150,200)
    
    return edges



img = cv.imread('Dataset/ISIC_0024397.jpg',0)
filtered_result = filterImage(img)
cvt.showImage("edges", filtered_result)
