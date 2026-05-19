set(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)) 
training_loader = torch.utils.data.DataLoader(simple_dataset)  

#global variables
epoch = 250 
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(2, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 2),
)



def train_simple_network(model,epoch,training_loader,device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(epoch):
        model.train() 
        total_loss = 0.0 

        for X,y in training_loader:
            X.to(device) 
            y.to(device)
            y_hat = model(X)
            loss = loss_function(y_hat,y)
            optimizer.zero_grad()
            loss.backward() 

            optimizer.step() 

            total_loss = total_loss+loss.item() 



train_simple_network(model,epoch,training_loader,device) 

def visualize2DSoftmax(X, y, model, title=None):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20),
    np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = torch.nn.functional.softmax(logits, dim=1).numpy()
    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(20,20),
    levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    if title is not None:
       ax.set_title(title)
visualize2DSoftmax(X, y, model)







    
    


