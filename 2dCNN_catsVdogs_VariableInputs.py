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


train_set= ImageFolder(root='/Users/rayhanrazzak/Desktop/PetImages', transform=ToTensor() ) #variable tensor sizes
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.output_num = [4,2,1]
        self.conv1=nn.Conv2d(in_channels=3, out_channels = 6, kernel_size = 5, bias = False) #in_channels = 3 because RGB
        self.conv2=nn.Conv2d(in_channels=6, out_channels = 12, kernel_size =5, bias = False)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60 )
        self.out=nn.Linear(in_features = 60, out_features = 2)
    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = F.leaky_relu(t)

        t = self.conv2(t)
        #t = F.leaky_relu(t)
        #t = SPPLayer(num_levels=1) #TypeError: spatial_pyramid_pool() missing 1 required positional argument: 'out_pool_size'
        spp = spatial_pyramid_pool(t,1,[int(t.size(2)),int(t.size(3))],self.output_num)

        #print(spp.shape)

        fc1 = self.fc1(spp)
        fc1 = F.leaky_relu(fc1)

        fc2 = self.fc2(fc1)
        fc2 = F.leaky_relu(fc2)

        fc2 = self.out(fc2)
        s = nn.Sigmoid()
        output = s(fc2)
        return output

network = Network()
train_loader  = torch.utils.data.DataLoader(train_set, shuffle=True)
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)


for epoch in range(1):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    print('Epoch: ', epoch, 'Total Correct: ', total_correct, 'Loss: ', total_loss)
print(total_correct/len(train_set))
