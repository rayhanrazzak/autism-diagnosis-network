import numpy as np
import os
import nibabel as nib
from dipy.align.reslice import reslice
from dipy.data import get_fnames


positive_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive"
negative_dir = "D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls"
new_dir = ""
k = 0
for i in os.listdir(negative_dir):
    directory = os.path.join(negative_dir,i)
    #print(directory)
    img = nib.load(directory)
    data = img.get_data()

    affine = img.affine
    zooms = img.header.get_zooms()[:3]
    #print(zooms)
    new_zooms = (1.0,1.0,1.0)
    data2, affine2, = reslice(data,affine, zooms, new_zooms)
    img2 = nib.Nifti1Image(data2,affine2)

    #print("D:/voxel corrected/anatomical/negative/{}".format(i))

    nib.save(img2, "D:/voxel corrected/anatomical/negative/{}".format(i))
