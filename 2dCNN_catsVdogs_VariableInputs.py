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


train_set= ImageFolder(root='D:/PetImages/', transform=ToTensor() ) #variable tensor sizes
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
'''
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.output_num = [1,12,50,50]
        self.conv1=nn.Conv2d(in_channels=3, out_channels = 6, bias = False) #in_channels = 3 because RGB
        self.conv2=nn.Conv2d(in_channels=6, out_channels = 12, bias = False)
        self.pool = nn.AdaptiveAvgPool2d(5)
        self.fc1 = nn.Linear(in_features = 25*12, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60 )
        self.out=nn.Linear(in_features = 60, out_features = 2)

    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = F.leaky_relu(t)

        t = self.conv2(t)

        #TypeError: spatial_pyramid_pool() missing 1 required positional argument: 'out_pool_size'
        spp = spatial_pyramid_pool(self=None,previous_conv=t,num_sample=1,previous_conv_size=[int(t.size(2)), int(t.size(3))], out_pool_size=self.output_num)
        print(spp.shape)
        #spp = spp.view(spp.shape[0],-1)
        #print(spp.shape)
        #print(spp.shape)
        fc1 = self.fc1(spp)
        fc1 = F.leaky_relu(fc1)

        fc2 = self.fc2(fc1)
        fc2 = F.leaky_relu(fc2)

        fc2 = self.out(fc2)

        return fc2'''
class Network(nn.Module):
    '''
    A CNN model which adds spp layer so that we can input multi-size tensor
    '''
    def __init__(self, input_nc=3, ndf=64,  gpu_ids=[]):
        super(Network, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = [4,2,1]

        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(10752,4096)
        self.fc2 = nn.Linear(4096,2)

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv3(x)
        x = F.leaky_relu(self.BN2(x))

        x = self.conv4(x)
        # x = F.leaky_relu(self.BN3(x))
        # x = self.conv5(x)
        spp = spatial_pyramid_pool(self = None, previous_conv=x,num_sample=1,previous_conv_size=[int(x.size(2)),int(x.size(3))],out_pool_size=self.output_num)
        # print(spp.size())
        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)
        '''s = nn.Sigmoid()
        output = s(fc2)'''
        return fc2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = Network()
network = network.to(device)
train_loader  = torch.utils.data.DataLoader(train_set, shuffle=True)
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)


for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        #images = images.view(images.shape[0], -1)
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
