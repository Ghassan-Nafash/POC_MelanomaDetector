import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import os


# Gamma correction
def display_img(title, input_img):
    columns = 6
    rows = 2
    fig = plt.figure(title, figsize=(24, 6))
    axi = ["Original Image", "Gamma Correction",
           "Blurred Image", "gray_input", "Binary Image","external_contours", "with_ocular_detection", "irrigular_border"]
    for i in range(1, columns*rows-3):
        ax = fig.add_subplot(rows, columns,i)
        if i >= 5:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1], cmap='gray')
            
        else:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1])

    fig.suptitle(title)
    plt.show()


def threshold(gray_input):
    #kernal = np.ones((5,5),np.uint8)
    thr, thresh_img = cv2.threshold(gray_input, 120, gray_input.max(), cv2.THRESH_OTSU)
    #thr, thresh_img = cv2.threshold(gray_input, 50, gray_input.max(), cv2.THRESH_OTSU) # Anna Testing

    noise = closing(thresh_img)

    thresh_img2 = cv2.adaptiveThreshold(noise, gray_input.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
    
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

def detect_border_irruglarity(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) > 4:
        # Fitting an Ellipse
        ellipse = cv2.fitEllipse(cnt)
        ellipse_cnt = cv2.ellipse2Poly( (int(ellipse[0][0]),int(ellipse[0][1]) ) ,( int(ellipse[1][0]),int(ellipse[1][1]) ),int(ellipse[2]),0,360,1)
        comp = cv2.matchShapes(cnt, ellipse_cnt, 1, 0.0)
        print(comp)
        cv2.ellipse(image,ellipse,(255,255,255),2)

    cv2.drawContours(image, cnt, -1, (255,255,255), 3)

    return image


def preproc_img():
    img_index = list(range(306, 314))  # 556 to index the last image
    for i in img_index:
        img_path = '../POC_MelanomaDetector/Dataset/ISIC_0024{}.jpg'.format(i)
        original_img = cv2.imread(img_path).astype(np.float32) / 255
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        gamma_img = np.array(gamma_correction(original_img_rgb, gamma=1.3)*255).astype(np.uint8)
        blurred_img = blur(gamma_img)
        gray_input = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
        bin_img = threshold(gray_input)

        external_contours = find_contours(bin_img, original_img_rgb, ocular_detection=False).astype(np.uint8)

        external_contours_with_ocular_detection = find_contours(bin_img, original_img_rgb, ocular_detection=True).astype(np.uint8)
        # detecting border irruglarity
        image_with_irrugular_border = detect_border_irruglarity(external_contours_with_ocular_detection)
        display_img("Image " + str(i), [original_img_rgb, gamma_img, blurred_img, gray_input, bin_img, 
                        external_contours, external_contours_with_ocular_detection, image_with_irrugular_border])
    

    return bin_img


def find_contours(input_img, original_img, ocular_detection=True):
    contour, hierarchy = cv2.findContours(input_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros(input_img.shape)
    longest_cntr = None
    largest_area = 0
    count = 0
    for cntr in contour:
        area = cv2.contourArea(cntr)
        if area > largest_area:
            is_artificial = True
            if ocular_detection:
                # Check if the current contour is a artificial circle (dermatological microscope objective) or natural shape
                is_artificial = is_artificial_circle(cntr, input_img.shape)
            if not is_artificial or not ocular_detection:        
                largest_area = area
                longest_cntr = cntr
    if not longest_cntr is None:
        cv2.drawContours(external_contours,[longest_cntr],-1,255,2)

    return external_contours


def is_artificial_circle(contour, shape):
    """
    Check if the current contour is a artificial circle (dermatological microscope objective) or natural shape
    Using only Hough transform
        Based on parameters: n of edge points, accum sum of Hough 
        detects if the given edge is an artificial artefact (lense ocular) or natural skin feature
    """
    detected = False
    mask = np.zeros(shape)
    cv2.drawContours(mask, contour,-1,1,2)

    # Hough circle detection with scikit-image
    hough_radii = [330]
    hough_res = sk.transform.hough_circle(mask, hough_radii)

    accums, cx, cy, radii = sk.transform.hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1, normalize=False)
    non_zero = np.count_nonzero(mask)

    if accums[0]/non_zero > 0.005/100 and non_zero > 100:

        detected = True

    return detected


def main():
    preproc_img()
    if cv2.waitKey(1) & 0xFF == ord('q'):

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()