"""
This script is responsible for Segmentation
"""
import cv2
import numpy as np
from skimage import data, segmentation, color
from skimage.future.graph import rag_mean_color, cut_threshold
from matplotlib import pyplot as plt


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


if __name__ == "__main__":
    img_index = list(range(422,425))  # 556 to index the last image
    
    for i in img_index:
        # img_path = "C:/Users/ancik/Můj disk/TU_BS/1 WS TUBS/Digitale Bildverarbeitung/Projekt DBV/Dataset/ISIC_0029{}.jpg".format(i)
        img_path = "C:/Users/ancik/Documents/GitHub/Dataset/ISIC_0029{}.jpg".format(i)
        # img_path = '../POC_MelanomaDetector/Dataset/ISIC_0029{}.jpg'.format(i)
        orig_img = cv2.imread(img_path).astype(np.float32) / 255
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        img = orig_img

        labels1 = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
        out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

        g = rag_mean_color(img, labels1)
        labels2 = cut_threshold(labels1, g, 29)
        out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                            figsize=(6, 8))

        ax[0].imshow(out1)
        ax[1].imshow(out2)

        for a in ax:
            a.axis('off')

        plt.tight_layout()