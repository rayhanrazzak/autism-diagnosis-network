import os
import numpy as np
import SimpleITK as sitk


root_dir = 'd:/voxel corrected/anatomical'
positive_dir = '{}/positive'.format(root_dir)
negative_dir = '{}/negative'.format(root_dir)

for i in os.listdir(positive_dir):
    directory = '{}/{}'.format(positive_dir,i)
    readable_image = sitk.ReadImage(directory)
    single_array = sitk.GetArrayFromImage(readable_image)
    print(np.shape(single_array))
for l in os.listdir(negative_dir):
    directory = '{}/{}'.format(negative_dir,i)
    readable_image = sitk.ReadImage(directory)
    single_array = sitk.GetArrayFromImage(readable_image)
    print(np.shape(single_array))
