import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(title, input_img):
    columns = 4
    rows = 2
    fig = plt.figure(figsize=(24, 6))
    axi = ["Original Image", "Blurred Image", "Binary Image",
           "external_contours", "Region Of Interest"]
    for i in range(1, columns*rows - 2):
        ax = fig.add_subplot(rows, columns, i)
        if i == 3 or i == 4:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1], cmap='gray')

        else:
            ax.set_title(axi[i-1])
            ax.imshow(input_img[i-1])
    fig.suptitle(title)
    plt.show()


def gamma_correction(input_img, gamma):
    # rase every pixel to the power of gamma ,the larger gamma the darker the image is
    input_img = np.array(input_img/255.0)
    result = np.power(input_img, 1/gamma) * 255
    result = np.array(result).astype(np.uint8)
    return result


def Histogram(input_img):
    # Contrast limited Adaptive Histogramm Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
    clahe_img = clahe.apply(input_img)
    plt.imshow(clahe_img, cmap='gray')
    plt.title("Contrast limited Adaptive Histogram Equalization")
    plt.show()
    return clahe_img


def blur(input_img):
    # 10 Diameter of the neighborhood, sigma color 50, sigma space 50
    blured = cv2.bilateralFilter(input_img, 25, 60, 60)
    # blured = cv2.GaussianBlur(input_img,(5,5),sigmaX=10,sigmaY=10)
    # blured = cv2.medianBlur(input_img,7)
    return blured


def closing(input_img):
    kernal = np.ones((5, 5), dtype=np.uint8)
    foreground_remove = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernal, iterations=3)
    # plt.imshow(foreground_remove)
    # plt.show()
    return foreground_remove


def opening(input_img):
    kernal = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernal, iterations=2)
    # plt.imshow(opened)
    # plt.show()
    return opened


def threshold(input_img):
    gray_input = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray_input,cmap='gray')
    # plt.show()
    # eq_input = Histogram(gray_input)
    # plt.hist(eq_input,bins=100, range=(0,255))
    # plt.show()
    # print(gray_input.dtype)
    # normalized Otsu
    norm_img = cv2.normalize(gray_input, None, 0, 255, cv2.NORM_MINMAX)
    # plt.imshow(norm_img,cmap='gray')
    # plt.title("Norm_img")
    # plt.show()
    # print(norm_img.min(),norm_img.max())
    thr, thresh_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(thresh_img,cmap='gray')
    # plt.title("thresh_img")
    # plt.show()
    # noise = closing(thresh_img)
    # plt.imshow(noise,cmap='gray')
    # plt.show()
    noise = closing(thresh_img)
    # plt.title("Noise Close")
    # plt.imshow(noise,cmap='gray')
    # plt.show()
    noise_open = opening(noise)
    # opened = opening(noise)
    # plt.imshow(noise_open,cmap='gray')
    # plt.title("Noise Open")
    # plt.show()

    thresh_img2 = cv2.adaptiveThreshold(noise_open, gray_input.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    # plt.imshow(thresh_img2,cmap='gray')
    # plt.show()
    return thresh_img2



def find_contours(input_img):
    global mask
    mask = np.zeros_like(input_img)
    contour, hierarchy = cv2.findContours(input_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('hierarchy', hierarchy)
    external_contours = np.zeros(input_img.shape)
    # internal_contours = np.zeros(input_img.shape)
    longest_cntr = None
    largest_area = 0
    for cntr in contour:
        area = cv2.contourArea(cntr)    
        if area > largest_area:
            largest_area = area
            longest_cntr = cntr
    _,_,w,h = cv2.boundingRect(longest_cntr)
    rect_area = w*h
    extent = float(largest_area)/rect_area
    if extent > 0.1:
        print("Extent", extent)
        
        cv2.drawContours(external_contours, [longest_cntr], -1, 255, 2)
        mask = cv2.drawContours(mask, [longest_cntr], -1, 255, -1)
        
        M = cv2.moments(longest_cntr)
        if M["m00"] != 0:
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # print("Center of Mass",cx , cy)
            
            major_diameter = 2 * np.sqrt(M["mu20"] / M["m00"])
            minor_diameter = 2 * np.sqrt(M["mu02"] / M["m00"])
            
        perimeter = cv2.arcLength(longest_cntr,True)
        
        
        ir_A = perimeter / largest_area #Asymmetry
        Circularity = perimeter**2 / largest_area #Circularity
        ir = perimeter**2 / (4*largest_area*np.pi) #Irregularity
        ir_ab = (4*largest_area*np.pi)/perimeter #Abnormality
        
        # Circularity_indx = (4*largest_area*np.pi) / perimeter**2
        
        
        # ir_B = perimeter / major_diameter
        
        # ir_C = perimeter * (1/minor_diameter - 1/major_diameter)
        
        # ir_D = major_diameter - minor_diameter
        
        print("Asymmetry",  ir_A,"\n Circularity",Circularity,"\n Irregularity",ir,"\n Abnormality",ir_ab)

        # mu11 = M['mu11']
        # mu02 = M['mu02']
        # mu20 = M['mu20']
        # a = np.sqrt(mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4*mu11**2))
        # b = np.sqrt(mu20 + mu02 - np.sqrt((mu20 - mu02)**2 + 4*mu11**2))
        # a_1 = 2 * np.maximum(a, b)
        # b_1 = 2 * np.minimum(a, b)
        
        print("Major diameter: ", major_diameter*2)
        print("Minor diameter: ", minor_diameter*2)
        # print("A: ", a_1)
        # print("B: ", b_1)
        ellipse = cv2.fitEllipse(longest_cntr)
        print("Maximum Axis, Minimum Axis", ellipse[1])
        
        # cv2.ellipse(external_contours,ellipse,255,2)
        cv2.circle(external_contours, (cx, cy), 5, 255, -1)
   
    else:
        cv2.putText(external_contours,"Unable to detect closed contours!",(50,250),fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)
    
    # cv2.line(external_contours,farthest_points[0],farthest_points[1],255,1)
    # epsilon =cv2.arcLength(longest_cntr,True)
    # approx = cv2.approxPolyDP(longest_cntr,epsilon,True)
    # print("longest_cntr: ",longest_cntr)
    # plt.imshow(input_img,cmap='gray')
    # plt.show()
    return external_contours


def edges(input_img):

    # this formula help us find a good values for the the threshold paramerter
    # formula steps:
    #               1. calculate the median pixel value
    #               2. set a lower bound & upper bound threshold

    # 1st
    med_val = np.median(input_img)
    print("Median of the inout image in the edges Function", med_val)
    # 2ed
    # set the lower threshold to either 0 or 70% of the median value whichever is Greater
    lower = int(max(0, 0.7*med_val))
    # set the upper threshold to either 130% of the median or the max 255, whichever is smaller
    upper = int(min(255, 1.3*med_val))
    edge = cv2.Canny(image=input_img, threshold1=lower, threshold2=upper)
    plt.title("EDGE")
    plt.imshow(edge, cmap='gray')
    plt.show()
    return edge


# TODO mask the roi using to color spaces HSV and Ycrcb
"""
H: 0-20, S: 20-255, V: 20-255 (for dark brown or black melanoma)
H: 20-40, S: 20-255, V: 20-255 (for blue or gray melanoma)
H: 0-180, S: 30-255, V: 30-255 (for melanoma of any color)
(pink)
Hue: 300-330
Saturation: 30-100
Value: 30-100
"""


def color_specs(input_img):
    global mask
    roi = np.zeros_like(input_img)
    roi[mask == 255] = input_img[mask == 255]
    roi = cv2.bitwise_and(roi, input_img, mask=mask)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    
    mask_1 = cv2.inRange(hsv_roi, np.array([0, 20, 20]), np.array([20, 255, 255]))  # Dark Brown or black
    mask_res1 =  cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_1)
    
    
    mask_2 = cv2.inRange(hsv_roi, np.array([20, 20, 20]), np.array([40, 255, 255])) # Blue or Gray
    mask_res2 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_2)
    
    res1 = cv2.bitwise_or(mask_res1,mask_res2)
    
    mask_3 = cv2.inRange(hsv_roi, np.array([150, 30, 30]), np.array([180, 255, 255])) # pink
    mask_res3 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_3)
    
    
    mask_4 = cv2.inRange(hsv_roi, np.array([272,63,54]), np.array([282,75,92])) # purple
    mask_res4 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_4)
    
    res2 = cv2.bitwise_or(mask_res3,mask_res4)
    
    final_res = cv2.bitwise_or(res1,res2)
    
    
    
    return final_res


def preproc_img():
    img_index = list(range(370, 500))  # 556 to index the last image
    for i in img_index:
        img_path = 'POC_MelanomaDetector-main/Dataset/ISIC_0024{}.jpg'.format(
            i)
        original_img = cv2.imread(img_path)
        assert original_img is not None, "file could not be read, check with os.path.exists()"
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # print(original_img.max(), original_img.min())
        # eq_img = Histogram(original_img)
        gamma_img = np.array(gamma_correction(original_img, gamma=0.85))  # choose gamma value
        blurred_img = blur(gamma_img)
        # canny_edg = edges(blurred_img)
        bin_img = threshold(blurred_img)
        # bin_img = DEM(bin_img)
        # edge = edges(bin_img)
        external_contours = find_contours(bin_img)  # .astype(np.uint8)
        # external_contours = DEM(external_contours)
        roi = color_specs(gamma_img)
        display_img("Image " + str(i),[original_img, blurred_img, bin_img, external_contours, roi])


def main():
    preproc_img()
    if cv2.waitKey(1) & 0xFF == ord('q'):

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
