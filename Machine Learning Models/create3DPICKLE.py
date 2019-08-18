import pickle
import SimpleITK as sitk
import numpy as np
train_dir = 'd:/voxel corrected/anatomical'
negative_dir = '{}/negative'.format(train_dir)
positive_dir = '{}/positive'.format(train_dir)
X_tensor = []
positive_label = [1,0]
#positive_label = torch.FloatTensor(positive_label)
negative_label = [0,1]
#negative_label = torch.FloatTensor(negative_label)
y_tensor = []

for i in os.listdir(negative_dir):
    directory = '{}/{}'.format(negative_dir, i)
    readable_image = sitk.ReadImage(directory)
    single_array = sitk.GetArrayFromImage(readable_image)
    single_array = np.interp(single_array, (single_array.min(), single_array.max()), (0,1)) #normalize data
    single_array = (single_array - single_array.mean())/single_array.std()
    #single_array = torch.from_numpy(single_array)
    X_tensor.append(single_array)
    y_tensor.append(negative_label)

for j in os.listdir(positive_dir):
    directory = '{}/{}'.format(positive_dir, j)
    readable_image = sitk.ReadImage(directory)
    single_array = sitk.GetArrayFromImage(readable_image)
    single_array = np.interp(single_array, (single_array.min(), single_array.max()), (0,1)) #normalize data
    single_array = (single_array - single_array.mean())/single_array.std() #standardize data
    #single_array = torch.from_numpy(single_array)
    X_tensor.append(single_array)
    y_tensor.append(positive_label)

final_list = []
for i in range(220):
    item = [ X_tensor[i], y_tensor[i] ]
    final_list.append(item)

with open("final_list.pkl", "wb") as f:
    pickle.dump(final_list,f)
