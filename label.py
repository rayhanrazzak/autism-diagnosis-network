from torchvision.datasets import ImageFolder
import torch
from torchvision.transforms import ToTensor
data = ImageFolder(root='/Users/rayhanrazzak/Desktop/PetImages', transform=ToTensor())

print(data.classes)
x,y = data[0]
from torch.utils.data import DataLoader
#train_loader = torch.utils.data.DataLoader(data)
train_loader = torch.utils.data.DataLoader(data, batch_size = 1000, shuffle=True)

for batch in train_loader:
    images, labels = batch
    print(labels)
#k = 0  number of dogs
#l = 0 number of cats
'''
for x, y in train_loader:
    print(x) # image
    print(y) # image label
    if y == [0,1]:
        k += 1
    if y == [1,0]:
        l += 1
'''
