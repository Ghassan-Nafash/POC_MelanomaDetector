import numpy as np

import cv2
import os, os.path
from matplotlib import pyplot as plt


# NOTES
# python version and compatibility with opencv (or with Pycharm?)
# functions from openCVTools


def load_images(n_images=10, path="./Dataset/"):
    """
    Loads the required number of images from the dataset
    :param path:
    :param n_images: required n of images
    :return: list of images
    """
    imgs = []
    n = 0
    for f in os.listdir(path):
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        n += 1
        if n >= n_images:
            break
    return imgs


def load_image_name(img_name, path="./Dataset/"):
    """
    Loads specific image from dataset
    :param path:
    :param n_images: required n of images
    :return: list of images
    """
    img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_images(images, target_h=450, target_w=600):
    resized = []
    for i in range(len(images)):
        img = images[i]
        s = img.shape
        h = s[0]
        w = s[1]
        if h != target_h or w != target_w:
            dim = (target_w, target_h)
            resized.append(cv2.resize(img, dim, interpolation=cv2.INTER_AREA))
        else:
            resized.append(img)
    return resized


def normalise_images(images):
    """
    normalise list of images
    :param images:
    :return:
    """
    normalised = []
    for i in range(len(images)):
        img = images[i]

        outlier_fraction = 0
        imgF = img.astype(np.float32)
        cv2.normalize(imgF, imgF, 0 - outlier_fraction, 1 + outlier_fraction, cv2.NORM_MINMAX)
        if outlier_fraction == 0:
            return imgF
        imgF = np.clip(imgF, 0, 1)

        normalised.append(imgF)
        return normalised


if __name__ == '__main__':
    n_images = 2
    # images = load_images(n_images=n_images)
    images = load_image_name("ISIC_0024342" + ".jpg")
    resized = resize_images(images, target_h=450, target_w=600)
    normalised = normalise_images(images)

    # comparing images
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    # rows = int(np.ceil(n_images / columns))
    rows = 1
    for i in range(rows * columns):
        if i % 3 == 0:
            img = images[int(i/3)]
            title = "orig"
        elif i % 3 == 1:
            img = resized[int(i/3)]
            title = "resized"
        else:
            img = normalised[int(i/3)]
            title = "normalised"
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
        plt.title(title)
    plt.show()
