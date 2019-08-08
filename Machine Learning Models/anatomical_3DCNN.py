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

# ---------create data loader, optimizer, accuracy test, CUDA device------------

train_set =

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU device for cuda
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01) #create optimizer
train_loader  = torch.utils.data.DataLoader(train_set, shuffle=True) #create dataloader (train_set is the tensor)

# ----------------------------Define Model--------------------------------------

class Network(nn.Module): #neural network class
    def __init__(self, input_nc = , ndf = ): #initialized variables
        super(Network, self).__init__()

        self.output_num = [x,y,z]
        self.conv1 = nn.Conv3d(input_nc, ndf,' x, y, z,' bias = False)
        self.conv2 = nn.Conv3d(ndf, ndf * 2, 'x,y,z,' bias = False)

    def forward(self,x): #defining forward pass

        #conv layer
        #relu
        #conv layer
        #batch normalization layer
        #relu
        #conv layer
        #spp layer
        #fully connected layer
        #fully connected layer
        #return

        #


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
# ------------------------------------------------------------------------------
