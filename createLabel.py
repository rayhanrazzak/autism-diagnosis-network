import pandas as pd
import csv
import os
'''
k = 117
file_directory = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive'
for i in os.listdir(file_directory):
    file_path = '{}/{}'.format(file_directory,i)
    print(file_path)

    os.rename(str(file_path), 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive/mprage{}.nii.gz'.format(k)  )
    k = k + 1
    '''


ASD_id_list = []
ASD_file = [0] * 104
control_id_list = []

id_list = []
file_directory = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive'

def makeList(list, files_directory):
    q = 0
    for i in os.listdir(files_directory):
        ASD_file[q] = str(i) #figure out how to create new items in list
        q = q + 1

makeList(ASD_id_list, file_directory)
#print(ASD_file)
#new_file = ASD_file
dirdir = 'D:/ABIDE 1/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls'

next_file = [0] * 116

q = 0
for i in os.listdir(dirdir):
    next_file[q] = str(i)
    q = q + 1
#print(next_file)

next_file.extend(ASD_file)
print(next_file)

print(len(next_file))
'''
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

'''
#id = control_list + autism_list
#label = label_list + labels
rows = zip(next_file)
with open('anat_label.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
'''

#print(autism_list)

'''
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
'''
# createLabel('/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of autism positive')
# createLabel('/D:/ABIDE 1/raw MRI/raw MRIs (functional and anatomical)/ABIDE/anatomical/nii files of controls')



'''
