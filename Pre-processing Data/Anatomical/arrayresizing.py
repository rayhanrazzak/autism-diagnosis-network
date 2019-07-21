import numpy as np
import nibabel as nib
import itertools
import os

negative_dir = "D:/voxel corrected/anatomical/negative"
positive_dir = "D:/voxel corrected/anatomical/positive"
new_dir = "D:/voxel corrected/size corrected/anatomical/negative"
new_x = 50
new_y = 50
new_z = 50

for i in os.listdir(positive_dir):
    #print(i)
    path = os.path.join(positive_dir, i)
    #print(path)
    original_file = nib.load(path).get_data()
    shape_array = original_file.shape
    x_size = shape_array[0]
    y_size = shape_array[1]
    z_size = shape_array[2]
    delta_x = x_size/new_x
    delta_y = y_size/new_y
    delta_z = z_size/new_z

    new_data = np.zeros((new_x,new_y,new_z))

    for x, y, z in itertools.product(range(new_x), #credit: https://github.com/Yt-trium/nii-resize/blob/master/nii-resize.py
                                     range(new_y),
                                     range(new_z)):
        new_data[x][y][z] = original_file[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]


    img = nib.Nifti1Image(new_data, np.eye(4))
    img.to_filename("{}/{}".format(new_dir,i))
