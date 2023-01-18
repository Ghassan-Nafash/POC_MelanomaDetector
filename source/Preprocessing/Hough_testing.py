import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks


data = np.array([[136, 0.0128, 1], 
                [1367, 0.103, 1], 
                [170, 0.0176, 1], 
                [65, 0.00694, 1], 
                [306, 0.0320, 1], 
                [383, 0.3632, 1], 
                [589, 0.0299, 1], 
                [261, 0.011, 1],
                [261, 0.011, 1], 
                [306, 0.0299, 1], 
                [964, 0.003, 1], 
                [1517, 0.008, 1],
                [309, 0.009, 1], 
                [683, 0.010, 1], 
                [532, 0.0389, 1], 
                [822, 0.055, 1], 
                [65, 0.005, 0], 
                [106, 0.00854, 0], 
                [119, 0.009, 0], 
                [2411, 0.0267, 0], 
                [24, 0.0267, 0], 
                [42, 0.0058, 0],
                [1253, 0.02617, 0], 
                [2310, 0.022435, 0],
                [45, 0.011, 0],
                [48, 0.005, 0],
                [35, 0.012, 0], 
                [98, 0.007, 0]])

x = data[:, [0,1]]
y = data[:, -1].astype(int)

plt.scatter(x[:, 0][y==0], x[:, 1][y==0], s=3, c='r')
plt.scatter(x[:,0][y==1], x[:,1][y==1], s=3, c='b')
plt.xlabel('n of edge points')
plt.ylabel('Hough accum value')
plt.legend(['skin', 'occular'])
plt.show()


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
