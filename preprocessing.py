import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# gamma correction (already done in our datasets)
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

# image bgr to rgb
img = cv2.imread("test5.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize image(already done in our datasets)
img = cv2.resize(img, (600, 450), interpolation = cv2.INTER_AREA)
gammaImg = gammaCorrection(img, 2)

# median filter，ksize = 9
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
hist_value, = plt.plot(hist)
plt.xlim([0,256])

# get x, y value of the histogram
x = hist_value.get_xdata()
y = hist_value.get_ydata()
p = [None] * 256
for i in range(0, 256):
    p[i] = np.interp(i, x, y)
sum_all = sum(p)
h_max = 0
s_max = 0
s = 0
np.seterr(divide = 'ignore')
while(s < 256):
    p_b = 0
    for i in range(0, s+1):
        p_b = p_b + p[i]/sum_all

    if p_b > 0:
        h_b = 0
        b_control = 0
        j = 0
        while(j < s+1):
            b_control = (p[j] / sum_all) / p_b * np.log2((p[j] / sum_all) / p_b)
            if (math.isnan(b_control) == False):
                h_b = h_b + b_control

            # print(j, h_b, b_control)
            j = j + 1
        h_b = -h_b
        h_w = 0
        w_control = 0
        k = s+1
        while(k < 256):
        # for k in range(s+1, 256):
            w_control = (p[k] / sum_all) / (1 - p_b) * np.log2((p[k] / sum_all) / (1 - p_b))
            if(math.isnan(w_control) == False):
                h_w = h_w + w_control
            k = k + 1
        h_w = -h_w
        h = h_b + h_w
        #print(h, h_max, s_max, p[s])
        #print(h, p[104], p_b, h_w, h_b)
        if h >= h_max:
            h_max = h
            s_max = s
        #if s == 104:
            #print(h, p[104], p_b, h_w, h_b)
    s = s + 1
# print(s_max, h_max)
np.seterr(divide = 'warn')


# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()

# chose threshold
# ???
# how to get the threshold automatically
# ???
# adaptive
ret, img_threshold = cv2.threshold(intensity, s_max, 255, cv2.THRESH_BINARY)

# closing, opening, dilation, canny edges, get the boundary and add it to the origin image
kernel2 = np.ones((5, 5), np.uint8)
kernel3 = np.ones((7, 7), np.uint8)
img_closing = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel3)

img_opening = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel3)

img_dilation2 = cv2.dilate(img_opening,kernel2,iterations = 1)

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
plt.title('threshold =  %d' %s_max), plt.xticks([]), plt.yticks([])
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
