# ------------------------------imports-----------------------------------------
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from spp_layer import spatial_pyramid_pool
from niftidataset import *
import nibabel as nib
import SimpleITK as sitk
import os
import numpy as np

# ---------create data loader, optimizer, accuracy test, CUDA device------------
'Urgent: Create train_set of n-dimensional MRIs for dataloader'

# create a list like this:
# train_set = [[[input], [label]]], each index has two lists with in (MRI values + label)
# train_set = ToTensor(train_set)
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
    #single_array = torch.from_numpy(single_array)
    X_tensor.append(single_array)
    y_tensor.append(negative_label)

for j in os.listdir(positive_dir):
    directory = '{}/{}'.format(positive_dir, j)
    readable_image = sitk.ReadImage(directory)
    single_array = sitk.GetArrayFromImage(readable_image)
    #single_array = torch.from_numpy(single_array)
    X_tensor.append(single_array)
    y_tensor.append(positive_label)

final_list = []
for i in range(220):
    item = [ X_tensor[i], y_tensor[i] ]
    final_list.append(item)
#print(final_list[0])
#print(X_tensor[0].size(0))
#print(y_tensor[0].size(0))
'''
print(len(X_tensor)) # should return 220
print(len(y_tensor)) # should return 220
print(X_tensor[0])
print(y_tensor[0]) # should return [0,1]'''
#y_tensor = np.asarray(y_tensor, dtype='float32')
#X_tensor = np.asarray(X_tensor, dtype=object)
#y_tensor = np.asarray(y_tensor, dtype=np.float32)
#y_tensor = y_tensor.astype('float32')


#print(X_tensor[0])
#print(y_tensor[0])
#print(X_tensor[200])
#print(y_tensor[200])


#X_tensor = torch.stack([torch.Tensor(i) for i in X_tensor]) # error because each item has a different size (same dimension # though)
#y_tensor = torch.stack([torch.Tensor(i) for i in y_tensor]) #no problems with y_tensor because each index is the same size

#train_set = torch.utils.data.TensorDataset(X_tensor, y_tensor)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU device for cuda
#train_loader  = torch.utils.data.DataLoader(train_set, shuffle=True) #create dataloader (train_set is the tensor)

# ----------------------------Define Model--------------------------------------

class Network(nn.Module): #neural network class
    def __init__(self, input_nc = 1, ndf = 64, gpu_ids = [], num_class = 2): #initialized variables
        super(Network, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = (4,2,2,1)
        self.conv1 = nn.Conv3d(input_nc, ndf, 4, 2, 2, 1, bias = False)

        self.conv2 = nn.Conv3d(ndf, ndf * 2, 4, 2, 1, 1, bias = False)
        self.BN1 = nn.BatchNorm3d(ndf*2)
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, 4, 1, 1, 1, bias = False)

        self.fc1 = nn.Linear(32000,4096)
        self.fc2 = nn.Linear(4096,num_class)
        self.pool = nn.AdaptiveAvgPool3d(5)
    def forward(self,x): #defining forward pass
        #identity
        x = x
        print('Original shape of x:' ,x.shape)


        #conv layer
        x = self.conv1(x)

        #relu
        x = F.leaky_relu(x)

        #conv layer
        x = self.conv2(x)

        #batch normalization layer
        x = self.BN1(x)

        #relu
        x = F.leaky_relu(x)

        #conv layer
        x = self.conv3(x)

        print('Shape of x after conv3', x.shape)
        x = self.pool(x)
        print('Shape of x after pool', x.shape)
        #return x
        x = x.view(x.shape[0],-1)
        print("Shape of x after view", x.shape)
        fc1 = self.fc1(x)
        fc2 = self.fc2(fc1)
        #fc2 = fc2.reshape(2)
        print('Shape of fc2', fc2.shape)
        print(fc2)
        s = nn.Sigmoid()
        output = s(fc2)
        print('This is the output:', output)
        return output

        #spp layer
'''    spp = spatial_pyramid_pool(self = None, previous_conv=x,num_sample=1,previous_conv_size=[int(x.size(2)),int(x.size(3)), int(x.size(4))],out_pool_size=self.output_num)
        print(spp.size())
'''
        #fully connected layer


'''
        fc1 = self.fc1(x)

        #fully connected layer
        fc2 = self.fc2(fc1)
        print(fc2.shape)
        #return
        return fc2
'''

# ------------------Create instance of model------------------------------------

network = Network() #initialize instance of neural network
#network = network.to(device) #move neural network from CPU to GPU
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01) #create optimizer

# -------------------train neural network---------------------------------------
for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in final_list:
        images, labels = batch
        images = np.reshape(images,(1,1,np.size(images,0), np.size(images,1), np.size(images,2)))
        images = images.astype('float32')
        images = torch.from_numpy(images)
        #print(images[0])

        labels = np.asarray(labels).reshape(1,2)
        print("Labels dtype:", labels.dtype)
        print("Labels dtype:", labels.dtype)
        labels = torch.from_numpy(labels)
        labels = labels.long()
    #    labels = labels.squeeze()
        labels =torch.argmax(labels)
        print(labels)
        print('Shape of label:', labels.shape)
        #images = images.to(device)
        #labels = labels.to(device)
        preds = network(images)
        #preds = preds.to(device)
        print('this is the preds:', preds)
        print('this is the shape of preds: ', preds.shape, 'this is the shape of squeezed labels: ', labels.shape)
        loss = F.cross_entropy(preds, labels.unsqueeze(0))
        print('this is the loss:', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    print('Epoch: ', epoch, 'Total Correct: ', total_correct, 'Loss: ', total_loss)
print(total_correct/len(train_set))

# ------------------------------end---------------------------------------------
