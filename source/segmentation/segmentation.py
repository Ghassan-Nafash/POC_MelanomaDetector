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
    def load(path: str, range_start: int, range_end: int) -> dict:
        """
        returns loaded images in RGB format as a dictionary 
            - dict keys are image numbers from the dataset (last 5 digits)
        """
        img_index = list(range(range_start,range_end+1))
        images = dict()
        for i in img_index:
            img_path = path + "ISIC_00" + str(i) + ".jpg"
            img = cv2. imread(img_path).astype(np.float32) / 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[i] = img
        return images


    def load_all(path: str):
        """
        returns loaded images in RGB format as a dictionary 
            - dict keys are image numbers from the dataset (last 5 digits)
        """
        images = dict()
        valid_ext = [".jpg"]
        for f in os.listdir(path):
            filename = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_ext:
                i = int(filename[-6:-1])
                img = cv2. imread(os.path.join(path, f)).astype(np.float32) / 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[i] = img
        return images


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

        # resulting level set
        ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                    smoothing=3)

        bin_img = MorphACWE.level_set_to_binary(ls)
        return bin_img
    

    def level_set_to_binary(level_set_image):
        binary_image = np.zeros(level_set_image.shape)
        binary_image[level_set_image > 0] = 1
        return binary_image


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


if __name__ == "__main__":
    start_index = 29422
    end_index = 29429
    images = Segmentation.load("C:/Users/ancik/Documents/GitHub/Dataset/", 29422, 29429)
    # images = Segmentation.load_all("C:/Users/ancik/Documents/GitHub/Dataset/")
    for img_number in images.keys():
        img = images[img_number]
        seg = MorphACWE.segment(img)
        MorphACWE.display(img, seg)

