import os
#print(os.getcwd())

os.chdir('D:')
#file_dir = 'D:/ABIDE 1/anatomical/autism positive'
#print(file_dir)


def oneFolderNewDir(file_dir,autism_true): 
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
                #os.chdir("D:/ABIDE 1/anatomical/autism positive/KKI_50791/KKI_50791/anat/NIfTI")
                #for l in os.listdir(os.curdir):
                #    print(l)
                file_path = '{}/anat/NIfTI/mprage.nii.gz'.format(directory)
                #print(file_path)
                #new_path = 'D:/ABIDE 1/anatomical/nii files of autism positive/mprage{}.nii.gz'.format(k)
                #print(new_path)
                k = k + 1
                if(autism_true == 'true'):
                    os.rename('{}'.format(file_path),'D:/ABIDE 1/anatomical/nii files of autism positive/mprage{}.nii.gz'.format(k))
                if(autism_true == 'false'):
                    os.rename('{}'.format(file_path), "D:/ABIDE 1/anatomical/nii files of controls/mprage{}.nii.gz".format(k))

#oneFolderNewDir('D:/ABIDE 1/anatomical/autism positive', 'true')
#oneFolderNewDir('D:/ABIDE 1/anatomical/controls', 'false')
