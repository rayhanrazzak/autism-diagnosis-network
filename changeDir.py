import os
print(os.getcwd())

file_dir = 'D:/ABIDE 1/anatomical/autism positive'
#print(file_dir)
k = 0

for i in os.listdir(file_dir):
    if not i.startswith('.'):
        #print(i)
        new_dir = '{}/{}'.format(file_dir,i)
        #print(new_dir)
        #print(os.listdir(new_dir))
        for l in os.listdir(new_dir):
            #print(l)
            directory = '{}/{}'.format(new_dir,l)
            #print(directory)
            file_path = '{}/anat/NIfTI/mprage.nii'.format(directory)
            #print(file_path)
            #print('{}'.format(file_path))
            #print('D:/ABIDE 1/anatomical/nii files of autism positive/mprage{}.nii'.format(k))
            k = k + 1
            os.rename('{}'.format(file_path),'D:/ABIDE 1/anatomical/nii files of autism positive/mprage{}.nii'.format(k))

            #for q in os.listdir(l):
        #    print(l)
        #if not i.startswith('.'):
            #print(os.listdir(i))
            #for l in os.listdir(i):
            #    print(l)
        #        k = 0
                #os.move('anat/NIfTI/mprage.nii', '/ABIDE 1/anatomical/autism positive/mprage{}.nii'.format(k))
            #    k = k + 1
