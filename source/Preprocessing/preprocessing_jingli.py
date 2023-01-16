import cv2
import numpy as np
from matplotlib import pyplot as plt

# gamma correction (already done in our datasets)
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

# image bgr to rgb
img = cv2.imread("Dataset/ISIC_0024320.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize image(already done in our datasets)
img = cv2.resize(img, (600, 450), interpolation = cv2.INTER_AREA)
gammaImg = gammaCorrection(img, 2)


# median filter，ksize=7 (in paper is 9)
median_img = cv2.medianBlur(img, 7)

# canny edges
edges = cv2.Canny(median_img, 100,200)

# dilation
# Taking a matrix of size 3 as the kernel
kernel = np.ones((3, 3), np.uint8)
img_dilation = cv2.dilate(edges, kernel, iterations=1)

# add together
img_dilation = cv2.merge((img_dilation,img_dilation,img_dilation))
img_add = cv2.add(img, img_dilation)

# rgb and intensity calculation(in paper: intensity = 1/3 * (b + g + r))
b, g, r = cv2.split(img_add)
b = 1/3 * b
g = 1/3 * g
r = 1/3 * r
intensity = b + g + r
intensity = intensity.astype(np.uint8)

# intensity histogram
hist = cv2.calcHist([intensity],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()

# chose threshold
# ???
# how to get the threshold automatically
# ???
# adaptive
ret, img_threshold = cv2.threshold(intensity, 120, 255, cv2.THRESH_BINARY)
#img_threshold = cv2.adaptiveThreshold(intensity, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#img_threshold = cv2.adaptiveThreshold(intensity, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# TODO: step Thicken morphological operation
# resource: A computer-aided diagnosis system for malignant melanomas


#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# closing, opening, dilation, canny edges, get the boundary and add it to the origin image
kernel2 = np.ones((5, 5), np.uint8)
kernel3 = np.ones((7, 7), np.uint8)
img_closing = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel3)

img_opening = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel3)

img_dilation2 = cv2.dilate(img_opening, kernel2, iterations = 1)

img_edges = cv2.Canny(img_dilation2, 100,200)

img_edges = cv2.merge((img_edges,img_edges,img_edges))
img_boundary = cv2.add(img, img_edges)

# show images
plt.subplot(341),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(342),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(343),plt.imshow(img_dilation,cmap = 'gray')
plt.title('Dilation'), plt.xticks([]), plt.yticks([])
plt.subplot(344),plt.imshow(img_add)
plt.title('Add together'), plt.xticks([]), plt.yticks([])
plt.subplot(345), plt.plot(hist)
plt.xlim([0,256])
plt.subplot(346),plt.imshow(img_threshold, cmap='gray')
plt.title('threshold = 104'), plt.xticks([]), plt.yticks([])
plt.subplot(347),plt.imshow(img_closing, cmap='gray')
plt.title('closing'), plt.xticks([]), plt.yticks([])
plt.subplot(348),plt.imshow(img_opening, cmap='gray')
plt.title('opening'), plt.xticks([]), plt.yticks([])
plt.subplot(349),plt.imshow(img_dilation2, cmap='gray')
plt.title('dilation again'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,10),plt.imshow(img_edges, cmap='gray')
plt.title('edge image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,11),plt.imshow(img_boundary)
plt.title('Add together'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,12),plt.imshow(gammaImg)
plt.title('gamma correction, gamma = 2'), plt.xticks([]), plt.yticks([])
plt.show()