from torchvision import datasets
from torchvision import transforms

data_path = '../../dataset'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

import matplotlib.pyplot as plt

len(cifar10)
plt.imshow(cifar10[99][0].permute(1,2,0))
plt.show()
