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

def eroding(input_img, k):
    kernel = np.ones((k, k), dtype=np.uint8)
    eroded = cv2.erode(input_img, kernel, iterations=1)

    return eroded


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
    # print("Largest area:", largest_area)
    a_score = 0
    b_score = 0
    if extent > 0.1:
        print("Result:")
        #print("Extent", extent)
        
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
        
        # get max and min diameter
        rect = cv2.fitEllipse(longest_cntr)
        (x, y) = rect[0]
        # print(rect[0])
        # print(rect[1])
        # print(rect[2])
        (weight, hight) = rect[1]
        angle = rect[2]

        if weight < hight:
            if angle < 90:
                angle -= 90
            else:
                angle += 90
        
        rows, cols = input_img.shape
        # print(rows,cols)
        rot = cv2.getRotationMatrix2D((x, y), angle, 1)
        # print(rot)
        cos = np.abs(rot[0, 0])
        sin = np.abs(rot[0, 1])
        nW = int((rows * sin) + (cols * cos))
        nH = int((rows * cos) + (cols * sin))

        rot[0, 2] += (nW / 2) - cols / 2
        rot[1, 2] += (nH / 2) - rows / 2

        warp_mask = cv2.warpAffine(input_img, rot, (nH, nW))

        cnts, hierarchy = cv2.findContours(warp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2. contourArea(c) for c in cnts]
        contour_m = cnts[np.argmax(areas)]
        xx, yy, nW, nH = cv2.boundingRect(contour_m)
        warp_mask = warp_mask[yy:yy + nH, xx:xx + nW]

        # test_ellipse = input_img.copy()
        # cv2.ellipse(test_ellipse, rect, 255, 3)

        # rmajor = max(weight, hight) / 2
        # xtop = x + cos*rmajor
        # ytop = y + sin*rmajor
        # xbot = x + cos*rmajor
        # ybot = y + (-sin)*rmajor
        # cv2.line(test_ellipse, (int(xtop),int(ytop)), (int(xbot),int(ybot)), 255, 3)

        # plt.imshow(test_ellipse)
        # plt.show()

        # get horizontal asymmetry
        # flip: A flag to specify how to flip the array; 
        # 0 means flipping around the x-axis and 
        # positive value (for example, 1) means flipping around y-axis
        flipContourHorizontal = cv2.flip(warp_mask, 1) # y-axis
        # plt.imshow(flipContourHorizontal)
        # plt.show()
        flipContourVertical = cv2.flip(warp_mask, 0) # x-axis

        diff_horizotal = cv2.compare(warp_mask, flipContourHorizontal, cv2.CV_8UC1)
        diff_vertical = cv2.compare(warp_mask, flipContourVertical, cv2.CV_8UC1)

        diff_horizotal = cv2.bitwise_not(diff_horizotal)
        diff_vertical = cv2.bitwise_not(diff_vertical)

        # returns the number of nonzero pixels in the array matrix
        h_asym = cv2.countNonZero(diff_horizotal)
        v_asym = cv2.countNonZero(diff_vertical)

        A1 = round(float(h_asym) / largest_area, 2)
        A2 = round(float(v_asym) / largest_area, 2)
        print(A1, A2)

        if (A1 > 0.18):
            print("image is Asymmetry around y axis")
            a_score += 1
        else:
            print("image is Symmetry around y axis")
        if (A2 > 0.18):
            print("image is Asymmetry around x axis")
            a_score += 1
        else:
            print("image is Symmetry around x axis")
        # print(A1, A2, a_score)


        ir_A = perimeter / largest_area #Asymmetry
        Circularity = perimeter**2 / largest_area #Circularity
        ir = perimeter**2 / (4*largest_area*np.pi) #Irregularity
        ir_ab = (4*largest_area*np.pi)/perimeter #Abnormality

        if(1.3<ir<1.5):
            b_score = 1
        if(1.5<=ir<1.6):
            b_score = 2
        if(1.6<=ir<1.8):
            b_score = 3
        if(1.8<=ir<2):
            b_score = 4
        if(2<=ir<2.2):
            b_score = 5
        if(2.2<=ir<2.5):
            b_score = 6
        if(2.5<=ir<3):
            b_score = 7
        if(ir>=3):
            b_score = 8
        # if (ir > 1.8): # value based on the papers
        #     print("Border is irregular, score is: ", b_score)
        # else:
        #     print("Border is regular, score is: ", b_score)
        
        # Circularity_indx = (4*largest_area*np.pi) / perimeter**2
        
        
        # ir_B = perimeter / major_diameter
        
        # ir_C = perimeter * (1/minor_diameter - 1/major_diameter)
        
        # ir_D = major_diameter - minor_diameter
        
        # print("Asymmetry",  ir_A,"\n Circularity",Circularity,"\n Irregularity",ir,"\n Abnormality",ir_ab)

        # mu11 = M['mu11']
        # mu02 = M['mu02']
        # mu20 = M['mu20']
        # a = np.sqrt(mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4*mu11**2))
        # b = np.sqrt(mu20 + mu02 - np.sqrt((mu20 - mu02)**2 + 4*mu11**2))
        # a_1 = 2 * np.maximum(a, b)
        # b_1 = 2 * np.minimum(a, b)
        
        #print("Major diameter: ", major_diameter*2)
        #print("Minor diameter: ", minor_diameter*2)
        # print("A: ", a_1)
        # print("B: ", b_1)
        ellipse = cv2.fitEllipse(longest_cntr)
        #print("Maximum Axis, Minimum Axis", ellipse[1])
        
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
    return external_contours, a_score, b_score

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
    imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    contours, hierarchy=cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    color = (0, 255, 0)
    for cntr in contours:
        area = cv2.contourArea(cntr) 
    result = cv2.drawContours(roi, contours, -1, color, 2)
    print("Area:", area)
    
    # image size: 450 * 600, x in range(0,450), y in range(0,600)
    num_white = 0
    num_black = 0
    num_red = 0
    num_light_brown = 0
    num_dark_brown = 0
    num_blue_gray = 0
    for x in range(0, 450):
        for y in range(0, 600):
            for c in contours:
                # point in the image? and if in ...
                point_in_image = cv2.pointPolygonTest(c, (x, y), False)

                if(point_in_image == 1): 
                    d_white = np.sqrt(((roi[x][y][0]-255)/255)**2 + ((roi[x][y][1] - 255)/255)**2 + ((roi[x][y][2] - 255)/255)**2)
                    if(d_white<0.5):
                        num_white += 1

                    d_black = np.sqrt(((roi[x][y][0]-0)/255)**2 + ((roi[x][y][1] - 0)/255)**2 + ((roi[x][y][2] - 0)/255)**2)
                    if(d_black<0.5):
                        num_black += 1

                    d_red = np.sqrt(((roi[x][y][0]-255)/255)**2 + ((roi[x][y][1] - 0)/255)**2 + ((roi[x][y][2] - 0)/255)**2)
                    if(d_red<0.5):
                        num_red += 1

                    d_light_brown = np.sqrt(((roi[x][y][0]-205)/255)**2 + ((roi[x][y][1] - 133)/255)**2 + ((roi[x][y][2] - 63)/255)**2)
                    if(d_light_brown<0.5):
                        num_light_brown += 1

                    d_dark_brown = np.sqrt(((roi[x][y][0]-101)/255)**2 + ((roi[x][y][1] - 67)/255)**2 + ((roi[x][y][2] - 33)/255)**2)
                    if(d_dark_brown<0.5):
                        num_dark_brown += 1

                    d_blue_gray = np.sqrt(((roi[x][y][0]-0)/255)**2 + ((roi[x][y][1] - 134)/255)**2 + ((roi[x][y][2] - 139)/255)**2)
                    if(d_blue_gray<0.5):
                        num_blue_gray += 1
    c_score = 0
    if((num_white / area) > 0.05):
        c_score += 1
        print("white")
    if((num_black / area) > 0.05):
        c_score += 1
        print("black")
    if((num_red / area) > 0.05):
        c_score += 1
        print("red")
    if((num_light_brown / area) > 0.05):
        c_score += 1
        print("light brown")
    if((num_dark_brown / area) > 0.05):
        c_score += 1
        print("dark brown")
    if((num_blue_gray / area) > 0.05):
        c_score += 1
        print("blue gray")

    print("c_score: ", c_score)
    # plt.imshow(roi)
    # plt.show()

    # hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    
    # mask_1 = cv2.inRange(hsv_roi, np.array([0, 20, 20]), np.array([20, 255, 255]))  # Dark Brown or black
    # mask_res1 =  cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_1)
    
    
    # mask_2 = cv2.inRange(hsv_roi, np.array([20, 20, 20]), np.array([40, 255, 255])) # Blue or Gray
    # mask_res2 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_2)
    
    # res1 = cv2.bitwise_or(mask_res1,mask_res2)
    
    # mask_3 = cv2.inRange(hsv_roi, np.array([150, 30, 30]), np.array([180, 255, 255])) # pink
    # mask_res3 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_3)
    
    
    # mask_4 = cv2.inRange(hsv_roi, np.array([272,63,54]), np.array([282,75,92])) # purple
    # mask_res4 = cv2.bitwise_and(hsv_roi,hsv_roi,mask=mask_4)
    
    # res2 = cv2.bitwise_or(mask_res3,mask_res4)
    
    # final_res = cv2.bitwise_or(res1,res2)
    
    
    
    return result, c_score

def diff_structures(input_img):
    global mask
    roi = np.zeros_like(input_img)
    roi[mask == 255] = input_img[mask == 255]
    roi = cv2.bitwise_and(roi, input_img, mask=mask)

    imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, black = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)

    # find contour and calculate the area
    contours, hierarchy=cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    color = (0, 255, 0)
    for cntr in contours:
        total_area = cv2.contourArea(cntr) 

    D_score = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get L channel in LAB color space
    imglab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L_channel, A_channel, B_channel = cv2.split(imglab)

    # adjusted image
    prd_adjusted = gamma_correction(L_channel, gamma=1.4)

    # thr, L_thr = cv2.threshold(prd_adjusted, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # image thresholding
    ret, prd_thr = cv2.threshold(prd_adjusted, 65, 255, cv2.THRESH_BINARY)
    ret, prd_thr_inv = cv2.threshold(prd_adjusted, 65, 255, cv2.THRESH_BINARY_INV)

    # find contour
    prd_contours, L_hierarchy=cv2.findContours(prd_thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Hi", L_hierarchy)
    L_roi = roi.copy()
    for i in range(len(prd_contours)):
        if(len(prd_contours) > 1):
            if L_hierarchy[0][i][3] != -1:
                prd_result = cv2.drawContours(L_roi, prd_contours, i, color, 1)
        else:
            prd_result = roi.copy()
            print("No irregular pigmented regions or dots detected!")
    
    prd_area = - total_area
    for i in prd_contours:
        area = cv2.contourArea(i)
        prd_area = area + prd_area
    
    if (prd_area / total_area >= 0.0015):
        D_score += 1
        cv2.putText(prd_result,'irregular pigmented regions or dots detected!',(20,200), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        print('irregular pigmented regions or dots detected!')
    
    print('prd_area of total area: ', prd_area / total_area)
    # print('d_score: ', D_score)
    
    # Result of irregular pigmented rogions or dots # 24310
    plt.subplot(231),plt.imshow(roi)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(L_channel, cmap="gray")
    plt.title('Image in L channel'), plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(prd_adjusted, cmap="gray")
    plt.title('Adjusted image'), plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(prd_thr, cmap="gray")
    plt.title('thesholding'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(prd_thr_inv, cmap="gray")
    plt.title('inverse of thesholding'), plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(prd_result)
    plt.title('Rusult'), plt.xticks([]), plt.yticks([])

    # get S channel in HSV
    imghsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H_channel, S_channel, V_channel = cv2.split(imghsv)

    bv_adjusted = gamma_correction(S_channel, gamma=1.2)
    ret, bv_thr = cv2.threshold(bv_adjusted, 85, 255, cv2.THRESH_BINARY)
    ret, bv_thr_inv = cv2.threshold(bv_adjusted, 85, 255, cv2.THRESH_BINARY_INV)

    # bv_opening = closing(bv_thr)
    contours, _ = cv2.findContours(bv_thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area)
        if area / total_area <= 0.002:
            cv_contours.append(contour)
        else:
            continue
            
    bv_copy = bv_thr.copy()
    bv_clear = cv2.fillPoly(bv_copy, cv_contours, (255, 255, 255))

    bv_contours, S_hierarchy=cv2.findContours(bv_clear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("s", S_hierarchy)

    bv_roi = roi.copy()

    bv_result = cv2.drawContours(bv_roi, bv_contours, -1, color, 2)

    bv_area = 0
    for cntr in bv_contours:
        area2 = cv2.contourArea(cntr)
        bv_area = area2 + bv_area
    bv_area = total_area - bv_area

    if (bv_area / total_area > 0.005):
        D_score += 1
        cv2.putText(bv_result,'Blue veil detected!',(100,200), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        print('Blue veil detected!')

    # # Result of Blue Veil # 24459
    # plt.subplot(231),plt.imshow(roi)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(232),plt.imshow(S_channel, cmap="gray")
    # plt.title('Image in S channel'), plt.xticks([]), plt.yticks([])
    # plt.subplot(233),plt.imshow(bv_adjusted, cmap="gray")
    # plt.title('Adjusted image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(234),plt.imshow(bv_thr, cmap="gray")
    # plt.title('thesholding'), plt.xticks([]), plt.yticks([])
    # plt.subplot(235),plt.imshow(bv_clear, cmap="gray")
    # plt.title('Image cleared'), plt.xticks([]), plt.yticks([])
    # plt.subplot(236),plt.imshow(bv_result)
    # plt.title('Rusult'), plt.xticks([]), plt.yticks([])

    # Structure less detecting
    sl_adjusted = cv2.GaussianBlur(S_channel, (5, 5), 0)

    #adapt_thresh = cv2.adaptiveThreshold(blurred, 255,
                                    #cv2.ADAPTIVE_THRESH_MEAN_C,
                                    #cv2.THRESH_BINARY_INV, 7, -2)
    #sl_adjusted = blur(S_channel)
    ret, sl_thr = cv2.threshold(sl_adjusted, 110, 255, cv2.THRESH_BINARY)
    sl_thr = cv2.bitwise_xor(sl_thr, black)
    # sl_thr_inv = cv2.bitwise_not(sl_thr)

    sl_eroded = eroding(sl_thr, 5)

    contours, _ = cv2.findContours(sl_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area)
        if area / total_area <= 0.002:
            cv_contours.append(contour)
        else:
            continue
            
    sl_copy = sl_eroded.copy()
    sl_clear = cv2.fillPoly(sl_copy, cv_contours, (0, 0, 0))

    #sl_clear = opening(sl_eroded)
    sl_contours, sl_hierarchy=cv2.findContours(sl_clear, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sl_roi = roi.copy()
    sl_result = cv2.drawContours(sl_roi, sl_contours, -1, color, 2)
    
    sl_area = 0
    for i in sl_contours:
        area3 = cv2.contourArea(i)
        sl_area = area3 + sl_area
    print(sl_area)
    
    if (sl_area / total_area > 0.005):
        D_score += 1
        cv2.putText(sl_result,'Structure less detected!',(100,200), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        print('Structure less detected!')

    # # Result of Structure Less # 24537
    # plt.subplot(231),plt.imshow(roi)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(232),plt.imshow(S_channel, cmap="gray")
    # plt.title('Image in S channel'), plt.xticks([]), plt.yticks([])
    # plt.subplot(233),plt.imshow(sl_thr, cmap="gray")
    # plt.title('thesholding'), plt.xticks([]), plt.yticks([])
    # plt.subplot(234),plt.imshow(sl_eroded, cmap="gray")
    # plt.title('image eroded'), plt.xticks([]), plt.yticks([])
    # plt.subplot(235),plt.imshow(sl_clear, cmap="gray")
    # plt.title('image cleared'), plt.xticks([]), plt.yticks([])
    # plt.subplot(236),plt.imshow(sl_result, cmap="gray")
    # plt.title('result'), plt.xticks([]), plt.yticks([])

    # YCbCr color space
    imgycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Y_channel, Cr_channel, Cb_channel = cv2.split(imgycrcb)
    Y_chn3 = cv2.merge((Y_channel, Y_channel, Y_channel))
    # test = L_channel + A_channel + B_channel

    reg_subtracted = cv2.subtract(Y_chn3, imglab)
    val1, val2, val3 = cv2.split(reg_subtracted)
    reg_gray = val1 + val2 + val3
    # print(reg_gray)

    ret, reg_thr = cv2.threshold(reg_gray, 70, 255, cv2.THRESH_BINARY)

    # # Result of Regression # 24459
    # plt.subplot(231),plt.imshow(roi)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(232),plt.imshow(reg_subtracted)
    # plt.title('Subtracted by Y channel and LAB'), plt.xticks([]), plt.yticks([])
    # plt.subplot(233),plt.imshow(Y_channel, cmap="gray")
    # plt.title('Y channel'), plt.xticks([]), plt.yticks([])
    # plt.subplot(234),plt.imshow(imgycrcb, cmap="gray")
    # plt.title('YCrCb'), plt.xticks([]), plt.yticks([])
    # plt.subplot(235),plt.imshow(reg_gray, cmap="gray")
    # plt.title('gray'), plt.xticks([]), plt.yticks([])
    # plt.subplot(236),plt.imshow(reg_thr, cmap="gray")
    # plt.title('test'), plt.xticks([]), plt.yticks([])

    # f = np.fft.fft2(imgray)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))

    # rows, cols = imgray.shape
    # crow,ccol = rows/2 , cols/2
    # # create a mask first, center square is 1, remaining all zeros
    # mask = np.zeros((rows,cols,2),np.uint8)
    # mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    # # apply mask and inverse DFT
    # fshift = fshift*mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = cv2.idft(f_ishift)
    # img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # plt.subplot(121),plt.imshow(imgray, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # plt.subplot(121),plt.imshow(imgray, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    return D_score


def preproc_img():
    img_index = list(range(310, 330))  # 556 to index the last image
    for i in img_index:

        # ISIC_0029{} 316, 318, 319, 343, 353, 363, 370, 397, 434, 453, 454,
        # 473, 480, 495, 502, 512, 513, 538, 547 are Melanoma
        # ISIC_0024{} 310, 313, 315, 323, 333, 351, 367, 400, 410, 449, 459,
        # 481, 482, 496, 516, 525, 537, 545, 546, 552, 554 are Melanoma

        img_path = '../Project/Dataset/ISIC_0024{}.jpg'.format(i)
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
        external_contours, a_score, b_score = find_contours(bin_img)  # .astype(np.uint8)
        # external_contours = DEM(external_contours)
        # longest_cntr = Contr(bin_img)
        # stencil = np.zeros(original_img.shape).astype(original_img.dtype)
        # color = [255, 255, 255]
        # cv2.fillPoly(stencil, longest_cntr, color)
        # result = cv2.bitwise_and(original_img, stencil)
        # plt.imshow(result)
        # plt.show()

        roi, c_score = color_specs(gamma_img)

        d_socre = diff_structures(gamma_img)

        TDS_score = a_score * 1.3 + b_score * 0.1 + c_score * 0.5 + d_socre * 0.5
        print("A: ", a_score, "B: ", b_score, "C: ", c_score, "D: ", d_socre, "TDS score is: ", TDS_score)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if(TDS_score > 5.45):
            cv2.putText(roi,'Melanoma!',(45,200), font, 3, (255, 0, 0), 4, cv2.LINE_AA)
        if(4.74 < TDS_score < 5.45):
            cv2.putText(roi,'Suspected melanoma!',(60,200), font, 1.3, (255, 0, 0), 2, cv2.LINE_AA)


        display_img("Image " + str(i),[original_img, blurred_img, bin_img, external_contours, roi])


def main():
    preproc_img()
    if cv2.waitKey(1) & 0xFF == ord('q'):

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()