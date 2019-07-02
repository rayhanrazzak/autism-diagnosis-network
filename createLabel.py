import pandas as pd
import csv
import os




control_list = []
label_list = []
k = 0
for i in os.listdir('/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls'):
    list1[k] = str(i)
    label_list[k] = 0
    k += 1
#print(control_list)



autism_list = []
labels = []
l = 0

for i in os.listdir('/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive'):
    list1[k] = str(i)
    labels[l] = 0
    l += 1


id = control_list + autism_list
label = label_list + labels
row = zip(id, label)
with open('anat_label.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


#print(autism_list)

'''



def createLabel(file_dir):
        if (file_dir == '/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive'): #for ASD positive (label = 1)
            for i in os.listdir(file_dir):
            # id = str(i)
            # label = 1
        elif (file_dir == '/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls'): #for controls (label = 0)
            for i in os.listdir(file_dir):
            # id = str(i)
            # label = 0
'''

# createLabel('/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive')
# createLabel('/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls')
