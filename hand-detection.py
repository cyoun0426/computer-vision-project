# Names:    Megha Devaraj, Christina Youn
# netIDs:   mdevaraj, cyoun
# File:     hand-detection.py

import cv2
import numpy as np
from skimage import measure
from sys import platform as sys_pf
from glob import glob
import os
import string
import warnings
warnings.filterwarnings('ignore')

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.plot()

# Global variables
Counter = 0

for letter in string.ascii_uppercase:
    images = glob('dataset/' + letter + '/*.jpg')
    newfolder = './images/' + letter + '/'
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)

    for i in images:
        # Read the image
        image = cv2.imread(i)                                 # CHANGE NAME OF FILE 1014, 2194
        image = cv2.resize(image, (500, 500))
        #cv2.imshow('Original image', image)

        # BGR image
        mask = cv2.inRange(image, np.array([22, 22, 58]), np.array([84, 83, 118]))
        #cv2.imshow('Mask image', mask)

        # Morphological processing
        kernel = np.ones((5,5), np.uint8)
        kernel2 = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel2, iterations=1)
        #cv2.imshow('After mophological operations', mask)

        # Find contours of objects
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find hand
        hand = 0
        maxarea = 0
        for c in range(len(contours)):
            tmp = cv2.contourArea(contours[c])
            if tmp > maxarea:
                maxarea = tmp
                hand = c
        x, y, w, h = cv2.boundingRect(contours[hand])
        cv2.rectangle(image, (x,y), (x+w, y+h), [0, 0, 255], 3)
        cv2.imwrite(newfolder + str(Counter) + '.jpg', image)

        Counter += 1
        print(Counter)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
