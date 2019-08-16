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

# ---------create data loader, optimizer, accuracy test, CUDA device------------
'Urgent: Create train_set of n-dimensional MRIs for dataloader'

# create a list like this:
# train_set = [[[input], [label]]], each index has two lists with in (MRI values + label)
# train_set = ToTensor(train_set)
train_dir = ''
train_set = NiftiDataset(train_dir, ToTensor())

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU device for cuda
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01) #create optimizer
train_loader  = torch.utils.data.DataLoader(train_set, shuffle=True) #create dataloader (train_set is the tensor)

# ----------------------------Define Model--------------------------------------

class Network(nn.Module): #neural network class
    def __init__(self, input_nc = 3, ndf = 64, gpu_ids = [], num_class = 2): #initialized variables
        super(Network, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = [4,2,2,1]
        self.conv1 = nn.Conv3d(input_nc, ndf, 4, 2, 2, 1, bias = False)

        self.conv2 = nn.Conv3d(ndf, ndf * 2, 4, 2, 1, 1, bias = False)
        self.BN1 = nn.BatchNorm3d(ndf*2)
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, 4, 1, 1, 1, bias = False)

        self.fc1 = nn.Linear(10752,4096)
        self.fc2 = nn.Linear(4096,num_class)

    def forward(self,x): #defining forward pass
        #identity
        x = x

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

        #spp layer
        spp = spatial_pyramid_pool(self = None, previous_conv=x,num_sample=1,previous_conv_size=[int(x.size(2)),int(x.size(3)), int(x.size(4))],out_pool_size=self.output_num)
        print(spp.size())

        #fully connected layer
        fc1 = self.fc1(spp)

        #fully connected layer
        fc2 = self.fc2(fc1)

        #return
        return fc2


# ------------------Create instance of model------------------------------------

network = Network() #initialize instance of neural network
network = network.to(device) #move neural network from CPU to GPU

# -------------------train neural network---------------------------------------
for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = network(images)
        preds = preds.to(device)
        loss = F.cross_entropy(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    print('Epoch: ', epoch, 'Total Correct: ', total_correct, 'Loss: ', total_loss)
print(total_correct/len(train_set))

# ------------------------------end---------------------------------------------
