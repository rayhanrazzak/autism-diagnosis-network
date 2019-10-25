# ------------------------------imports-----------------------------------------
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms
import random
from random import shuffle
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
import pickle

# ------------------open list that stores images, labels------------------------
with open("final_list.pkl", "rb") as f:
    final_list = pickle.load(f) # MRI array data and label array data
# -------------------------determine accuracy ----------------------------------
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
# ------------------------------ define GPU ------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU device for cuda

# ----------------------------Define Model--------------------------------------

class Network(nn.Module): #neural network class



    def __init__(self, input_nc = 1, ndf = 64, gpu_ids = [], num_class = 2): #initialized variables
        super(Network, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = (4,2,2,1)

# Use FIVE conv layers (conv, relu, maxpool)
        self.conv_layer1 = self._make_conv_layer(3, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)

# Use FIVE max pooling layers

# One 3D SPP layer
        self.spp_layer = spatial_pyramid_pool(self = None, previous_conv = x, num_sample = 1, previous_conv_size = [int(x.size(2)), int(x.size(3)), int(x.size(4))], out_pool_size = self.output_num)

# Two Fully Connected Layers
        self.fc1 = nn.Linear(4096,2048)
        self.fc2 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(256,num_class)

    def _make_conv_layer(self, in_c, out_c):

        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.relu(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.relu(),
        nn.MaxPool3d((2, 2, 2)),
        )

        return conv_layer


    def forward(self, x): #defining forward pass

        #identity
        x = x

        # Convolution Layers:
        x = self.conv_layer1(x)

        x = self.conv_layer2(x)

        x = self.conv_layer3(x)

        x = self.conv_layer4(x)

        print("This is the shape before spp: ", x.shape) #size that the input of fc1 needs to be

        #SPP Layer
        x = self.spp_layer()

        print("This is the shape before fc1: ", x.shape) #size that the input of fc1 needs to be

        #Fully Connected Layers:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #Second fully connected layer
        x = self.fc3(x)

        return x

# ------------------Create instance of model------------------------------------

network = Network() #initialize instance of neural network
network = network.to(device) #move neural network from CPU to GPU
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01) #create optimizer

# -------------------train neural network---------------------------------------
for epoch in range(5): #number of epochs
    total_loss = 0
    total_correct = 0
    random.shuffle(final_list) #shuffle = true for dataset

    for batch in final_list:
        images, labels = batch
        images = np.reshape(images,(1,1,np.size(images,0), np.size(images,1), np.size(images,2))) #reshape images array as [batch size, input_channels, x,y,z]
        images = images.astype('float32') # convert images array to float32 type
        images = torch.from_numpy(images) #convert images numpy array to torch tensor

        labels = np.asarray(labels) # convert labels list into a numpy array
        labels = torch.from_numpy(labels) #convert labels numpy array to torch tensor
        labels = labels.long() #convert dtype to long
        #labels =torch.argmax(labels) #take the greatest value and store it as a scalar
        images = images.to(device)
        labels = labels.to(device)
        preds = network(images)
        preds = preds.to(device)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    print('Epoch: ', epoch, 'Total Correct: ', total_correct, 'Loss: ', total_loss)
print(total_correct/len(train_set))

# ------------------------------end---------------------------------------------
