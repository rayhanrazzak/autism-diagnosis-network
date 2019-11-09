import torch
import torch.nn as nn
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
from fractions import gcd
from functools import reduce

#spatial_pyramid_pooling (spp) layer that uses greatest common divisors (GCD) to prevent padding
class spp:
    def __init__(self, x):
        self.x = x #tensor attribute
        self.size = np.shape(self.x) #size attribute
        self.a = [] #GCF of first dimension in self.path
        self.b = [] #GCF of second dimension in self.path
        self.c = [] #GCF of third dimension in self.path
    def pool(self, gcd): #MaxPools the tensor
        self.gcd = gcd
        self.out = nn.MaxPool3d((self.size[0] / self.gcd[0], self.size[1] / self.gcd[1], self.size[2] / self.gcd[2]), padding=False)
        return self.out
    def gcd(self, path): #generates the GCD of 3 dimensions for a path
        self.path = path
        for i in os.listdir(self.path):
            self.file = '{}/{}'.format(self.path, i)
            self.readImage = sitk.ReadImage(self.file)
            self.array = sitk.GetArrayFromImage(self.readImage)
            
            self.a.append(np.shape(self.array)[0]) #list of all files first dimension
            self.b.append(np.shape(self.array)[1])
            self.c.append(np.shape(self.array)[2])
        self.a_gcd = reduce(gcd, self.a)
        self.b_gcd = reduce(gcd, self.b)
        self.c_gcd = reduce(gcd, self.c)

        self.gcd = (self.a_gcd, self.b_gcd, self.c_gcd)
        return self.gcd #returns a tuple

''' How to run:
 
 Requirements:
    1. x is a tensor
    2. path is the directory of all MRIs
 
 spp_layer = spp(x) #creating instance of spp layer
 gcd = spp.gcd(path) #generating the GCD of each dimension of files in path
 spp_layer = spp.pool(gcd) #returning maxpool

'''
