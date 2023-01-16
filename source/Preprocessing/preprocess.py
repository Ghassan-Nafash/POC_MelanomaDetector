import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import os



# Gamma correction


def display_img(title, input_img):
    columns = 4
    rows = 2
    fig = plt.figure(title, figsize=(12, 6))
    axi = ["Original Image", "Gamma Correction",
           "Blurred Image", "Binary Image","external_contours", "with_ocular_detection"]
    for i in range(1, columns*rows -1):
        ax = fig.add_subplot(rows, columns,i)
        if i == 4 or i ==5 or i == 6:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1], cmap='gray')
            
        else:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1])

    fig.suptitle(title)
    plt.show()


def threshold(input_img):
    kernal = np.ones((5,5),np.uint8)
    gray_input = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    # print("size of the gray image= ",gray_input.shape)
    # print("max value of the gray image: ",gray_input.max())
    # plt.imshow(gray_input,cmap='gray')
    # plt.show()
    thr, thresh_img = cv2.threshold(gray_input, 120, gray_input.max(), cv2.THRESH_OTSU)
    # print("Otsu Thresh Value: ",thr)
    # plt.imshow(thresh_img,cmap='gray')
    # plt.show()
    noise = closing(thresh_img)
    # plt.imshow(noise,cmap='gray')
    # plt.show()
    # thresh_img2 = cv2.adaptiveThreshold(noise, gray_input.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_OTSU, 11, 11)
    
    thresh_img2 = cv2.adaptiveThreshold(noise, gray_input.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
    
    
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



def DEM(input_img):
    kern_dialate = np.ones((5,5), dtype=np.uint8)
    kern_erode = np.ones((5,5), dtype=np.uint8)
    result = cv2.erode(input_img,kern_erode,iterations=1)
    result = cv2.dilate(result,kern_dialate,iterations=1)
    #print(type(result))
    #print(result.shape)
    result = cv2.medianBlur(result,3)
    return result

def blur(input_img):
    # 10 Diameter of the neighborhood, sigma color 50, sigma space 50
    blured = cv2.bilateralFilter(input_img, 10,60,60)
    # blured = cv2.GaussianBlur(input_img,(5,5),sigmaX=10,sigmaY=10)
    return blured


def gamma_correction(input_img, gamma):
    # rase every pixel to the power of gamma ,the larger gamma the darker the image is
    result = np.power(input_img, gamma)
    return result


def preproc_img():
    # img_index = list(range(409, 410))  # 556 to index the last image
    ocular_images = os.listdir('../POC_MelanomaDetector/Dataset/ocular_dataset')
    # ocular_images = [24360, 24371, 24400, 24408, 24409, 24431, 24435, 24450, 24489, 24490, 24308] # a few images with ocular
    for i in ocular_images:
        img_path = '../POC_MelanomaDetector/Dataset/ocular_dataset/' + str(i)
        original_img = cv2.imread(img_path).astype(np.float32) / 255
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)


        # print(original_img.max(), original_img.min())
        gamma_img = np.array(gamma_correction(original_img, gamma=0.8)*255).astype(np.uint8)  # choose gamma value
        blurred_img = blur(gamma_img)
        bin_img = threshold(blurred_img)
        #bin_img = DEM(bin_img)
        external_contours = find_contours(bin_img, original_img, ocular_detection=False).astype(np.uint8)
        external_contours_with_ocular_detection = find_contours(bin_img, original_img, ocular_detection=True).astype(np.uint8)
        #external_contours = DEM(external_contours)
        external_contours_with_ocular_detection = find_contours(bin_img, original_img, ocular_detection=True).astype(np.uint8)
        display_img("Image " + str(i), [original_img, gamma_img, blurred_img, bin_img, external_contours, external_contours_with_ocular_detection])
    return bin_img


def find_contours(input_img, original_img, ocular_detection=True):
    
    contour, hierarchy = cv2.findContours(input_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print('hierarchy', hierarchy)
    external_contours = np.zeros(input_img.shape)
    #internal_contours = np.zeros(input_img.shape)
    longest_cntr = None
    largest_area = 0
    count = 0
    for cntr in contour:
        area = cv2.contourArea(cntr)
        if area > largest_area:
            is_artificial = True
            if ocular_detection:
                # TODO: Check if the current contour is a artificial circle (dermatological microscope objective) or natural shape
                is_artificial = is_artificial_circle(original_img, cntr, input_img.shape, draw=False)
            if not is_artificial or not ocular_detection:        
                largest_area = area
                longest_cntr = cntr
    # print("Area: ",largest_area)
    # print("longest_cntr: ",longest_cntr)
    if not longest_cntr is None:
        cv2.drawContours(external_contours,[longest_cntr],-1,255,2)
    """    for i in range(len(contour)):
        #external --> -1
        if hierarchy[0][i][3] ==-1:
            cv2.drawContours(external_contours,contour,i,255,2)
        #else:
            #cv2.drawContours(internal_contours,contour,i,255,2)
            """
    return external_contours


def is_artificial_circle(original_img, contour, shape, draw=False):
    """
    Check if the current contour is a artificial circle (dermatological microscope objective) or natural shape
    Using only Hough transform
        Based on parameters: n of edge points, accum sum of Hough 
        detects if the given edge is an artificial artefact (lense ocular) or natural skin feature
    """
    detected = False
    mask = np.zeros(shape)
    # mask = np.zeros(shape, dtype="uint8")
    cv2.drawContours(mask, contour,-1,1,2)

    # Hough circle detection with scikit-image
    # hough_radii = np.arange(330, 332, 1)
    hough_radii = [330]
    hough_res = sk.transform.hough_circle(mask, hough_radii)

    # Select the most prominent 1 circle
    accums, cx, cy, radii = sk.transform.hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1, normalize=False)
    non_zero = np.count_nonzero(mask)
    print(f"nonzero: {non_zero}")
    print(f"accums: {accums}")
    print("accums/non_zero: %.3f perc., radius: %d" %(accums[0]/non_zero * 100, radii[0] ))
    if accums[0]/non_zero > 0.005/100 and non_zero > 100:
        print(">>>>>> Occular detected <<<<<<<")
        detected = True

    # Draw circle
    if draw:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = sk.color.gray2rgb(mask)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = sk.draw.circle_perimeter(center_y, center_x, radius,
                                            shape=image.shape)
            image[circy, circx] = (220, 20, 20)
        image = np.clip(image, 0, 1)
        ax.imshow(image, cmap=plt.cm.gray)
        plt.show()
    return detected



def is_artificial_circle_02(original_img, contour, shape, draw=False):
    """
    NOT Finished
    Check if the current contour is a artificial circle (dermatological microscope objective) or natural shape
    Using HOUGH transform and COLOR information
    """
    detected = False
    mask = np.zeros(shape)
    # mask = np.zeros(shape, dtype="uint8")
    cv2.drawContours(mask, contour,-1,1,2)

    # average color on the contour 
    # hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    # masked = cv2.bitwise_and(original_img, original_img, mask=test_contour_img)
    # plt.imshow(masked)
    # plt.show()
    # hue, saturation, brightness, _ = cv2.mean(original_img, mask=mask)
    # print(f"h: {hue}, s: {saturation}, v: {brightness}")
    # print(f"brightness: {brightness}")

    # Hough circle detection with scikit-image
    # hough_radii = np.arange(330, 332, 1)
    hough_radii = [330]
    hough_res = sk.transform.hough_circle(mask, hough_radii)

    # Select the most prominent 1 circle
    accums, cx, cy, radii = sk.transform.hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1, normalize=False)
    non_zero = np.count_nonzero(mask)
    print(f"nonzero: {non_zero}")
    print(f"accums: {accums}")
    print("accums/non_zero: %.3f perc., radius: %d" %(accums[0]/non_zero * 100, radii[0] ))
    if accums[0]/non_zero > 0.005/100 and non_zero > 100:
        print(">>>>>> Occular detected <<<<<<<")
        detected = True

    # Draw circle
    if draw:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = sk.color.gray2rgb(mask)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = sk.draw.circle_perimeter(center_y, center_x, radius,
                                            shape=image.shape)
            image[circy, circx] = (220, 20, 20)
        image = np.clip(image, 0, 1)
        ax.imshow(image, cmap=plt.cm.gray)
        plt.show()
    return detected

# def findCircles(edgesImg, num=1):
#     """

#     """
#     circles = []
#     if num < 1:
#         return circles
#     thresh = 1
#     do = True
#     while do:
#         circles = cv2.HoughCircles(edgesImg, 
#         cv2.HOUGH_GRADIENT,
#         dp=1, 
#         minDist=1,
#         param2=thresh)
        
#         thresh += 1
#         print(f"Hough Thresh: {thresh}")
#         do = len(circles) > num
#     return circles, thresh

    
"""
    def edges():
        input_img = preproc_img()
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        blurred = cv2.GaussianBlur(input_img, (5, 5), sigmaX=10, sigmaY=10)
        # this formula help us find a good values for the the threshold paramerter
        # formula steps:
        #               1. calculate the median pixel value
        #               2. set a lower bound & upper bound threshold

        # 1st
        med_val = np.median(input_img)

        # 2ed

        # set the lower threshold to either 0 or 70% of the median value whichever is Greater
        lower = int(max(0, 0.7*med_val))
        # set the upper threshold to either 130% of the median or the max 255, whichever is smaller
        upper = int(min(255, 1.3*med_val))
        edge = cv2.Canny(image=blurred, threshold1=lower, threshold2=upper+100)
        plt.subplot(121)
        plt.imshow(edge, cmap='gray')
        plt.title("Detected Edges")
        plt.subplot(122)
        plt.imshow(input_img)
        plt.title("original Image")
        plt.show()
"""

def main():
    preproc_img()
    if cv2.waitKey(1) & 0xFF == ord('q'):

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
