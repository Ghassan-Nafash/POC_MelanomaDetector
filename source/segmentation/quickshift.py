"""
Testing script
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import sklearn 
from skimage.segmentation import quickshift, mark_boundaries

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, watershed


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
    img_index = list(range(422,423))  # 556 to index the last image
    for i in img_index:
        # img_path = "C:/Users/ancik/Můj disk/TU_BS/1 WS TUBS/Digitale Bildverarbeitung/Projekt DBV/Dataset/ISIC_0029{}.jpg".format(i)
        img_path = "C:/Users/ancik/Documents/GitHub/Dataset/ISIC_0029{}.jpg".format(i)
        # img_path = '../POC_MelanomaDetector/Dataset/ISIC_0029{}.jpg'.format(i)
        orig_img = cv2.imread(img_path).astype(np.float32) / 255
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        img = orig_img

        segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
        segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
                            start_label=1)
        segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
        gradient = sobel(rgb2gray(img))
        segments_watershed = watershed(gradient, markers=250, compactness=0.001)

        print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
        print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
        print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
        print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')

        # Displaying subplot with all the methods
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

        # ax[0, 0].imshow(mark_boundaries(img, segments_fz))
        # ax[0, 0].set_title("Felzenszwalbs's method")
        # ax[0, 1].imshow(mark_boundaries(img, segments_slic))
        # ax[0, 1].set_title('SLIC')
        # ax[1, 0].imshow(mark_boundaries(img, segments_quick))
        # ax[1, 0].set_title('Quickshift')
        # ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
        # ax[1, 1].set_title('Compact watershed')

        # for a in ax.ravel():
        #     a.set_axis_off()

        # plt.tight_layout()
        ## plt.savefig('./source/segmentation/output/' + 'ISIC_0029{}'.format(i) + "" + ".png")
        # plt.show()


        plt.imshow(mark_boundaries(img, segments_quick))
        plt.title('Quickshift ISIC_0029{}'.format(i))
        plt.axis('off')
        plt.savefig('./Outputs/' + 'Quickshift ISIC_0029{}'.format(i) + ".png")
        plt.show()


