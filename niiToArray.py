# test converting one anatomical MRI to an array
import os
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
#print(os.getcwd())
fMRI_dir = '/Users/Flash/Desktop/Caltech_51456/SCANS/rest/NIfTI/rest.nii'
fMRI_test_image = sitk.ReadImage(fMRI_dir)
fMRI_array = sitk.GetArrayFromImage(fMRI_test_image)
#print(np.shape(fMRI_array))
#print(fMRI_array)

anat_dir = '/Users/Flash/Desktop/Caltech_51456/SCANS/anat/NIfTI/mprage.nii'
anat_test_image = sitk.ReadImage(anat_dir)
anat_array = sitk.GetArrayFromImage(anat_test_image)
#print(np.shape(anat_array))
#print(anat_array)\
#print(os.getcwd())
file_dir = '/Users/Flash/Desktop/hello'
l = os.listdir(file_dir)
for q in l:
    #print(i)
    new_path = '{}/{}'.format(file_dir,q)
    #print(new_path)
    if not q.startswith('.'):
        for i in os.listdir(new_path):
            if not i.startswith('.'):
                #print(i)
                #print(file_dir)
                #print(q)
                print('{}/{}/{}'.format(file_dir,q,i))
                os.rename('{}/{}/{}'.format(file_dir,q,i), '/Users/Flash/Desktop/ye/{}'.format(i))
                #os.rename('{}/{}/{}'.format(file_dir,q,i), '/Users/Flash/Desktop/newfolder/')
    #new_path = '{0}'.format(i)
    #print(os.listdir(i))

#for i in os.listdir(file_dir):
    #print('{0}/{0}'.format(file_dir,i))
    #if '{0}/{0}'.format(file_dir,i):
