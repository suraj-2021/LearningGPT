""" I'm Creating a simple Neural Network With 2 hidden layers and 1 Activation Function """ 
import torch 
import numpy as np 
import seaborn as sns 
from torch.utils.data import Dataset,DataLoader 

#Creating Data Points 
X = np.linspace(0,20,num=200)
Y = X+ np.sin(X)*2+ np.random.normal(size=X.shape) 


#Crating Dataset 
class SimpleDataset(Dataset):
     def __init__(self,X,Y):
        super(SimpleDataset,self).__init__() 

        self.X = X.reshape(-1,1) 
        self.Y = Y.reshape(-1,1) 


     def __getitem__(self,index):
         return torch.tensor(self.X[index,:],dtype=torch.float32),torch.tensor(self.Y[index],dtype=torch.float32) 


     def __len__(self):

         return self.X.shape[0] 


#creating dataloader and other global variables
dataloader = DataLoader(SimpleDataset(X,Y)) 
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=1,out_features=10),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features=10,out_features=1) 

)


device = "cuda" if torch.cuda.is_available() else "cpu" 

epoch = 1000


def simple_neural_network(model,device,epoch):
    model.to(device) 
    loss_function = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001) 
    
    for _ in range(epoch):
        model.train()
        total_loss=0.0 

        for X,Y in dataloader:
            X = X.to(device) 
            Y = Y.to(device) 

            Y_hat = model(X) 
            optimizer.zero_grad()
            loss = loss_function(Y_hat,Y)
            loss.backward() 
            optimizer.step() 
            
            total_loss = total_loss+loss.item()


simple_neural_network(model,device,epoch)

with torch.no_grad():
   X = torch.tensor(X,dtype=torch.float32).cpu()
   X = X.reshape(-1,1) 
   Y_pred = model(X) 


sns.scatterplot(x = X.flatten(),y=Y.flatten(),color="blue") 

sns.lineplot(x=X.flatten(),y = Y_pred.flatten(),color="red")










        








