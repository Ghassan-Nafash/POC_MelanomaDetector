"""
This script is responsible for feature extraction
"""
import preprocess as prep
import numpy as np
# how to import from different dir? __init__.py

def find_pixel_diameter(contour):
    """
    finds the feature diameter in pixels 
    """


    return 0


if __name__ == "__main__":
    ocular_images = os.listdir('./Dataset/ocular_dataset')
    # ocular_images = [24360, 24371, 24400, 24408, 24409, 24431, 24435, 24450, 24489, 24490, 24308] # a few images with ocular
    for i in ocular_images:
        img_path = './Dataset/ocular_dataset/' + str(i) + ".jpg"
        original_img = cv2.imread(img_path).astype(np.float32) / 255
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # bin_img = prep.preproc_img()
    # cnt = prep.prefind_contours(bin_img, ocular_detection=True).astype(np.uint8)
    # prep.display_img("Image " , [bin_img, bin_img, bin_img, bin_img, bin_img, cnt, cnt])
    # diam = find_pixel_diameter(cnt)
