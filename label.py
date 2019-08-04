from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
data = ImageFolder(root='main_dir', transform=ToTensor())

print(data.classes)
x,y = data[0]
from torch.utils.data import DataLoader
loader = DataLoader(data)
#loader = DataLoader(data, shuffle=True)
for x, y in loader:
    print(x) # image
    print(y) # image label
