# Names:    Megha Devaraj, Christina Youn
# netIDs:   mdevaraj, cyoun
# File:     hand-detection.py

import cv2
import numpy as np
from skimage import measure
from sys import platform as sys_pf
import warnings
warnings.filterwarnings('ignore')

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.plot()

# Read the image
image = cv2.imread('A1014.jpg')                                 # CHANGE NAME OF FILE 1014, 2194
image = cv2.resize(image, (500, 500))
cv2.imshow('Original image', image)

# BGR image
mask = cv2.inRange(image, np.array([22, 22, 58]), np.array([84, 83, 118]))
cv2.imshow('Mask image', mask)

# Morphological processing
kernel = np.ones((5,5), np.uint8)
kernel2 = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel2)
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=kernel)
cv2.imshow('After mophological operations', mask)

# Final pre-processing
res = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Final", res)

#cc = cv2.connectedComponents(mask)
#ccimg = cc[1].astype(np.uint8)
#contour, hierarchy = cv2.findContours(ccimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.waitKey(0)
cv2.destroyAllWindows()
