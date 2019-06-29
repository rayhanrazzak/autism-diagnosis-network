# test converting one anatomical MRI to an array
import os
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
#print(np.eye(2))
file = '/Users/Flash/Desktop/Caltech_51456/SCANS/anat/NIfTI/mprage.nii'
test_image = sitk.ReadImage(file)


test_array = sitk.GetArrayFromImage(test_image)
print(test_array)
#os.chdir('/Desktop')
#print(os.getcwd())
