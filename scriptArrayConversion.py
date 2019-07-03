import SimpleITK as sitk
import numpy as np


anat_dir = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive'
control_dir = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls'


def MRI_dimensions(file_dir):
    for i in os.listdir(file_dir):
        readable_image = sitk.ReadImage(i)
        single_array = sitk.GetArrayFromImage(readable_image)
        print(np.shape(single_array))

MRI_dimensions(anat_dir)
MRI_dimensions(control_dir)
