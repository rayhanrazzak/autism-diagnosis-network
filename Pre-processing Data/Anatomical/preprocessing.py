from tqdm import tqdm
import numpy as np
from random import shuffle
import SimpleITK as sitk
import os
img_size = 50 #dimensions of the length and width dimensions

# [asd_positive, asd_negative]

positive_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive"

negative_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls"


def training_data():
    training_data = []
    for i in tqdm(os.listdir(positive_dir)):
        path = os.path.join(positive_dir, i)
        array = sitk.GetArrayFromImage(sitk.ReadImage(path))
        print(np.shape(array))

training_data()
