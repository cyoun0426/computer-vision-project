# Names:    Megha Devaraj, Christina Youn
# netIDs:   mdevaraj, cyoun
# File:     single-detection.py

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

def processImage(image):
    cv2.imshow('Original image', image)

    # BGR image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(image, np.array([0, 17, 135]), np.array([24, 90, 221]))
    mask2 = cv2.inRange(image, np.array([165, 0, 0]), np.array([180, 0, 0]))
    mask = mask1 + mask2
    cv2.imshow('Mask image', mask)

    # Morphological processing
    kernel = np.ones((5,5), np.uint8)
    kernel2 = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel2, iterations=1)
    cv2.imshow('After morphological operations', mask)

    return mask

def findHand(mask):
    # Find contour of hand
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hand = 0
    max = 0
    for c in range(len(contours)):
        tmp = cv2.contourArea(contours[c])
        if tmp > max:
            max = tmp
            hand = c
    x, y, w, h = cv2.boundingRect(contours[hand])
    cv2.rectangle(image, (x,y), (x+w, y+h), [0, 0, 255], 3)
    cv2.imshow('Found hand', image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return x, y, w, h

def cropImage(image, x, y, w, h):
    cv2.imshow('Before crop', image)
    cropimage = image[y-10:y+2*w, x-10:x+w+10]
    cv2.imshow("Cropped", cropimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropimage

def findFeatures(image):
    labels = measure.label(image)
    properties = measure.regionprops(labels)
    return properties

if __name__ == "__main__":
    imfile = './dataset/M/M1051.jpg'
    image = cv2.imread(imfile)
    image = cv2.resize(image, (500, 500))    
    mask = processImage(image)
    x, y, w, h = findHand(mask)
    crop = cropImage(mask, x, y, w, h)
