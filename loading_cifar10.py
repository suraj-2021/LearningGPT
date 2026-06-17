import torch 
from torch.utils.data import DataLoader 
import torchvision 
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns 
import tqdm.auto as tqdm


#load the CIFAR10 datasets from torchvision 
training_dataset = torchvision.datasets.CIFAR10("./data",train=True,download=True,transform=transforms.ToTensor())
testing_dataset = torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=transforms.ToTensor())

print(training_dataset[0].shape)

