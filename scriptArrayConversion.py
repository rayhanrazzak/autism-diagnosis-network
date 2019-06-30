import SimpleITK as sitk
import numpy as np


anat_dir = ""
for i in anat_dir:
    readable_image = sitk.ReadImage(i)
    single_array = sitk.GetArrayFromImage(readable_image)
