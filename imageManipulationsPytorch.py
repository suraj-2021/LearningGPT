import torch
import torchvision 
from torchvision import transforms
from matplotlib.pyplot import imshow 


x_train = torchvision.datasets.MNIST("./data",train=True,download=True,transform = transforms.ToTensor()) 

x_e1 = x_train[0] 
x_e2 = x_train[1] 
x_e3 = x_train[2] 


x_e1 = x_e1[0] 
x_e2 = x_e2[0]
x_e3 = x_e3[0]
#single image
x_color = torch.stack([x_e1[0,:],x_e2[0,:],x_e3[0,:]],dim=0) 

#change the color of the first image 

imshow(x_color.permute(1,2,0))


