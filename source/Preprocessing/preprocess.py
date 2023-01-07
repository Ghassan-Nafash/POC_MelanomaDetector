import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gamma correction


def display_img(input_img):
    columns = 4
    rows = 1
    fig = plt.figure(figsize=(24, 6))
    axi = ["Original Image", "Gamma Correction",
           "Blurred Image", "Binary Image"]
    for i in range(1, columns*rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        if i != 4:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1])
        else:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1], cmap='gray')

    plt.show()


def threshold(input_img):
    kernal = np.ones((5, 5), np.uint8)
    gray_input = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    # print("size of the gray image= ",gray_input.shape)
    # print("max value of the gray image: ",gray_input.max())
    # plt.imshow(gray_input,cmap='gray')
    # plt.show()
    thr, thresh_img = cv2.threshold(
        gray_input, 120, gray_input.max(), cv2.THRESH_OTSU)
    # print("Otsu Thresh Value: ",thr)
    # plt.imshow(thresh_img,cmap='gray')
    # plt.show()
    noise = closing(thresh_img)
    # plt.imshow(noise,cmap='gray')
    # plt.show()
    # thresh_img2 = cv2.adaptiveThreshold(noise, gray_input.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_OTSU, 11, 11)

    thresh_img2 = cv2.adaptiveThreshold(noise, gray_input.max(
    ), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 11)

    # mask = cv2.erode(thresh_img2, kernal, iterations = 2)
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(thresh_img2,cmap='gray')
    # plt.show()
    # result = cv2.addWeighted(src1=thresh_img,alpha=0.6,src2=thresh_img2,beta=0.4,gamma=0)
    # result = - thresh_img - thresh_img2
    return thresh_img2


def closing(input_img):
    kernal = np.ones((7, 7), dtype=np.uint8)
    foreground_remove = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernal)
    return foreground_remove


def opening(input_img):
    kernal = np.ones((3, 3), dtype=np.uint8)
    result = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernal)
    return result


def blur(input_img):
    # 10 Diameter of the neighborhood, sigma color 50, sigma space 50
    # blured = cv2.bilateralFilter(input_img, 10, 50, 50)
    blured = cv2.GaussianBlur(input_img, (5, 5), sigmaX=10, sigmaY=10)
    return blured


def gamma_correction(input_img, gamma):
    # rase every pixel to the power of gamma ,the larger gamma the darker the image is
    result = np.power(input_img, gamma)
    return result


def preproc_img():
    img_index = list(range(306, 308))  # 556 to index the last image
    for i in img_index:
        img_path = '../POC_MelanomaDetector/Dataset/ISIC_0024{}.jpg'.format(i)
        original_img = cv2.imread(img_path).astype(np.float32) / 255
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # print(original_img.max(), original_img.min())
        gamma_img = np.array(gamma_correction(original_img, gamma=0.8)*255).astype(np.uint8)  # choose gamma value
        blurred_img = blur(gamma_img)
        bin_img = threshold(blurred_img)

        display_img([original_img, gamma_img, blurred_img, bin_img])
    return bin_img


# def edges():
#     input_img = preproc_img()
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     blurred = cv2.GaussianBlur(input_img, (5, 5), sigmaX=10, sigmaY=10)
#     # this formula help us find a good values for the the threshold paramerter
#     # formula steps:
#     #               1. calculate the median pixel value
#     #               2. set a lower bound & upper bound threshold

#     # 1st
#     med_val = np.median(input_img)

#     # 2ed

#     # set the lower threshold to either 0 or 70% of the median value whichever is Greater
#     lower = int(max(0, 0.7*med_val))
#     # set the upper threshold to either 130% of the median or the max 255, whichever is smaller
#     upper = int(min(255, 1.3*med_val))
#     edge = cv2.Canny(image=blurred, threshold1=lower, threshold2=upper+100)
#     plt.subplot(121)
#     plt.imshow(edge, cmap='gray')
#     plt.title("Detected Edges")
#     plt.subplot(122)
#     plt.imshow(input_img)
#     plt.title("original Image")
#     plt.show()


def main():
    preproc_img()
    if cv2.waitKey(1) & 0xFF == ord('q'):

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
