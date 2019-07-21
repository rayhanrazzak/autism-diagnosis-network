from tqdm import tqdm
import numpy as np
from random import shuffle
import SimpleITK as sitk
import os
import cv2
img_size = 50 #dimensions of the length and width dimensions

# [asd_positive, asd_negative]

positive_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive"

negative_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls"

main_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical"
def training_data():
    k = 0
    training_data = []
    for i in os.listdir(positive_dir):
        path = os.path.join(positive_dir, i)
        array = sitk.GetArrayFromImage(sitk.ReadImage(path))
        #print(np.shape(array))
        label = [1,0]
        # resize the arrays here
        #training_data.append([np.array(array), np.array(label)])
        k += 1
        #print(k)
    for i in os.listdir(negative_dir):
        path = os.path.join(negative_dir, i)
        file = sitk.GetArrayFromImage(sitk.ReadImage(path))
        print(np.shape(array))
        hotarray = [0,1]
        # resize the arrays here
        #training_data.append([np.array(file), np.array(hotarray)])
    shuffle(training_data)
training_data()
