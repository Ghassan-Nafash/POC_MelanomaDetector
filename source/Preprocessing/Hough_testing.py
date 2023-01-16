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