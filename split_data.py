from random import shuffle
from glob import glob
import os
import shutil
import string
from itertools import islice
""" This file splits thee dataset folder into 60% training data, 20% testing data, and 20% validation data. """

filepath = "../computer-vision-project/dataset/"
data_types = ['training', 'testing', 'validation']
seclist = [1800,600,600] # based of 3000 total images

for letter in string.ascii_uppercase:
    print("Splitting " + letter + "...")
    images = glob(filepath + letter + "/*.jpg")

    # create new folders to store data
    for data_type in data_types: 
        newfolder = "../computer-vision-project/dataset/" + data_type + '/' + letter
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
    # separate images into lists
    shuffle(images)
    it = iter(images)
    sliced =[list(islice(it, 0, i)) for i in seclist]
    
    # move images into testing, training, validation folders
    i = 0
    for sublist in sliced:
        for image_path in sublist:
            shutil.move(image_path, "../computer-vision-project/dataset/" + data_types[i] + '/' + letter)
        i+=1  
    
    shutil.rmtree("../computer-vision-project/dataset/" + letter)

print("All done!")