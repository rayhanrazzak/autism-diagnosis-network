import os
file_dir = 'D:/ABIDE 1/anatomical/autism positive'
if not i.startswith('.'):
    for i in os.listdir(file_dir):
        if not i.startswith('.'):
            for l in os.listdir(i):
                k = 0
                os.move('anat/NIfTI/mprage.nii', '/ABIDE 1/anatomical/autism positive/mprage{}.nii'.format(k))
                k = k + 1
