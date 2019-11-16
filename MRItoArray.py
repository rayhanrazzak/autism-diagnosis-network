import SimpleITK as sitk
import numpy as np
import os
#what did you just delete no need to import numpy i used simpleITK instead
def MRI2Array(directory):
    for i in os.listdir(directory):
        dir = "{}/{}".format(directory,i)
        img = sitk.ReadImage(dir)
        array = sitk.GetArrayFromImage(img)
        #return array/np.amax(array)
        myMax = np.amax(array)
        array = array / myMax
        newMax = np.amax(array)
        print("this is the max value: ", newMax)
        print("It's index is ", np.where(array == np.max(array)))
        print("This is the size: ", array.shape)
    #return np.where(array == np.max(array))
    #return array

print(MRI2Array("D:/ABIDE1/minimally_preprocessed_fMRIs/Ages_6-10_With_ASD"))

