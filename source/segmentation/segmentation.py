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

    def display():
        """
        """
        pass

    @abstractclassmethod
    def segment(image):
        """
        To be implemented by inheriting classes.
        Returns contour
        """
        pass


class MorphACWE(Segmentation):
    """
    """
    __evolution = []

    def __store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """
        def _store(x):
            lst.append(np.copy(x))

        return _store

    def segment(image) -> np.array:
        """
        segments single image
        """
        image = rgb2gray(image)
        image = img_as_float(image)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        __evolution = []
        callback = MorphACWE.__store_evolution_in(__evolution)
        ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                    smoothing=3, iter_callback=callback)
        return ls
    
    def display(level_set):
        plt.imshow(level_set, cmap='gray')
        plt.axis('off')
        # plt.axis('flatten')
        plt.contour(level_set, [0.5], colors="r")
        plt.title("Morphological ACWE segmentation", fontsize=12)
        plt.show()

    def display_intermediate(level_set):
        """displays the intermediate steps of last execution of MorphACWE.segment"""
        plt.imshow(level_set, cmap="gray")
        plt.axis('off')
        contour = plt.contour()
        contour = plt.contour(MorphACWE.__evolution[2], [0.5], colors='g')
        contour.collections[0].set_label("Iteration 2")
        contour = plt.contour(MorphACWE.__evolution[7], [0.5], colors='y')
        contour.collections[0].set_label("Iteration 7")
        contour = plt.contour(MorphACWE.__evolution[-1], [0.5], colors='r')
        contour.collections[0].set_label("Iteration" + str(len(MorphACWE.__evolution)-1))
        plt.legend(loc="upper right")
        title = "Morphological ACWE evolution"
        plt.title(title, fontsize=12)
        plt.show()


if __name__ == "__main__":
    start_index = 29422
    end_index = 29429
    images = Segmentation.load("C:/Users/ancik/Documents/GitHub/Dataset/", 29422, 29429)
    for img_number in images.keys():
        img = images[img_number]
        seg = MorphACWE.segment(img)
        MorphACWE.display(seg)
        # MorphACWE.display_intermediate(seg)

