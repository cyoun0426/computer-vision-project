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
image = cv2.imread('A1014.jpg')                                 # CHANGE NAME OF FILE
cv2.imshow('Original image', cv2.resize(image, (500, 500)))

# BGR image
kernel = np.ones((5,5), np.uint8)
mask = cv2.inRange(image, np.array([7, 6, 40]), np.array([94, 102, 140]))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel)
cv2.imshow('After mophological operations', cv2.resize(mask, (500, 500)))
#cc = cv2.connectedComponents(mask)
#ccimg = cc[1].astype(np.uint8)
#contour, hierarchy = cv2.findContours(ccimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Convert image to HSV                                              FAILED!!!
#hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#hsv_image = hsv_image[:, :, 0]
#cv2.imshow('HSV image', cv2.resize(hsv_image, (500, 500)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
