"""
This script is responsible for Segmentation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import sklearn 
import skimage
from skimage.segmentation import quickshift, mark_boundaries

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, watershed

from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def display_img(input_img):
    columns = 5
    rows = 2
    fig = plt.figure(figsize=(24, 6))
    axi = ["Original Image", "Gamma Correction",
           "Blurred Image", "Binary Image","segmented", "external_contours"]
    for i in range(1, columns*rows -3):
        ax = fig.add_subplot(rows, columns,i)
        if i == 4 or i ==5:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1], cmap='gray')
            
        else:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1])
    plt.show()

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


if __name__ == "__main__":
    img_index = list(range(422,429))  # 556 to index the last image
    for i in img_index:
        # img_path = "C:/Users/ancik/Můj disk/TU_BS/1 WS TUBS/Digitale Bildverarbeitung/Projekt DBV/Dataset/ISIC_0029{}.jpg".format(i)
        img_path = "C:/Users/ancik/Documents/GitHub/Dataset/ISIC_0029{}.jpg".format(i)
        # img_path = '../POC_MelanomaDetector/Dataset/ISIC_0029{}.jpg'.format(i)
        orig_img = cv2.imread(img_path).astype(np.float32) / 255
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        img = rgb2gray(orig_img)

        # Morphological ACWE
        image = img_as_float(img)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                    smoothing=3, iter_callback=callback)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax = axes.flatten()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_axis_off()
        ax[0].contour(ls, [0.5], colors='r')
        ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

        ax[1].imshow(ls, cmap="gray")
        ax[1].set_axis_off()
        contour = ax[1].contour(evolution[2], [0.5], colors='g')
        contour.collections[0].set_label("Iteration 2")
        contour = ax[1].contour(evolution[7], [0.5], colors='y')
        contour.collections[0].set_label("Iteration 7")
        contour = ax[1].contour(evolution[-1], [0.5], colors='r')
        contour.collections[0].set_label("Iteration 35")
        ax[1].legend(loc="upper right")
        title = "Morphological ACWE evolution"
        ax[1].set_title(title, fontsize=12)



        fig.tight_layout()
        plt.savefig('./Outputs/' + 'Morfological ACWE ISIC_0029{}'.format(i) + ".png")
        plt.show()


