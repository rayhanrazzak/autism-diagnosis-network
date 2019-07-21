# example for converting MRI to an GetArrayFromImage
# prints out the dimensions of each MRI using the MRI_dimensions() function

import SimpleITK as sitk
import numpy as np
import os
import nibabel as nib

positive_dir = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive'
negative_dir = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls'


def MRI_dimensions(file_dir):
    for i in os.listdir(file_dir):
        #print(i)


        directory = '{}/{}'.format(file_dir,i)
        mriFile = nib.load(directory)
        print(mriFile.header.get_zooms()) #prints the voxel size for each MRI in the file directory
        readable_image = sitk.ReadImage(directory)
        single_array = sitk.GetArrayFromImage(readable_image)

        print(np.shape(single_array))

MRI_dimensions(positive_dir)

MRI_dimensions(negative_dir)
