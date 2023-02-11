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
    """
    Class for segmentation with Color filter method.
    To be implemented.
    """
    def segment(image) -> np.array:
        """
        To be implemented
        """
        pass
    
    def display(image, cont):
        """
        To be implemented
        """
        pass


class Thresholding(Segmentation):
    """
    Class for segmentation with Color filter method.
    To be implemented.
    """
    def segment(image) -> np.array:
        """
        To be implemented
        """
        pass
    
    def display(image, cont):
        """
        To be implemented
        """
        pass


class Improved_thresholding(Segmentation):
    """
    Class for segmentation with Color filter method.
    To be implemented.
    """
    def segment(image) -> np.array:
        """
        To be implemented
        """
        pass
    
    def display(image, cont):
        """
        To be implemented
        """
        pass







if __name__ == "__main__":
    start_index = 29422
    end_index = 29429
    images = Segmentation.load("C:/Users/ancik/Documents/GitHub/Dataset/", 29422, 29429)
    # images = Segmentation.load_all("C:/Users/ancik/Documents/GitHub/Dataset/")
    for img_number in images.keys():
        img = images[img_number]
        seg = MorphACWE.segment(img)
        MorphACWE.display(img, seg)

