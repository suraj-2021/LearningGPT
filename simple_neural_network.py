import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
import seaborn as sns 


#generate synthetic data 
X = np.linspace(0,20,num=200)
Y = X+np.sin(X)*2+np.random.normal(size=X.shape)

#Creating Dataset 

class SimpleDataset(Dataset):
  def __init__(self,X,Y):
    super(SimpleDataset,self).__init__()

    self.X = X.reshape(-1,1)
    self.Y = Y.reshape(-1,1) 

  def __getitem__(self,index):
      return torch.tensor(self.X[index],dtype=torch.float32), torch.tensor(self.Y[index],dtype=torch.float32)

  def __len__(self):

    return self.X.shape[0]


  
#creating dataloader that responds to the dataset 
dataloader = DataLoader(SimpleDataset(X,Y)) 
#creating gloabal variables to feed to the training function 

model = torch.nn.Linear(in_features=1, out_features=1) 
#creating device 
device = "cuda" if torch.cuda.is_available() else "cpu" 
#epoch variable 
epoch = 1000

def train_simple_network(model,device,dataloader,epoch):
      
      optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
      loss_function = torch.nn.MSELoss()

      for _ in range(epoch):
         model.train()

         total_loss =0.0 

         for X,Y in dataloader:
             X = X.to(device)
             Y = Y.to(device)
             
             y_hat = model(X)
             optimizer.zero_grad()

             loss = loss_function(y_hat,Y)

             loss.backward()

             optimizer.step() 

             total_loss=total_loss+1 


model.eval()
train_simple_network(model,device,dataloader,epoch)
with torch.no_grad():
      X = torch.tensor(X,dtype=torch.float32)
      X = X.reshape(-1,1)
      y_predictions = model(X)
sns.scatterplot(x=X.flatten(),y=Y.flatten())  
sns.lineplot(x=X.flatten(),y=y_predictions.flatten(),color='red')





       
      



