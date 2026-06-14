import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt # FIXED: Imported matplotlib to actually show the Seaborn plots
import torch
import torchvision 
from torchvision import transforms 
from torch.utils.data import DataLoader # FIXED: datasets -> data
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor()) 

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Global variables 
C = 1 
K = 16 
k_size = 3
D = 28 * 28
classes = 10

# Score functions 
score_funcs = {
    "accuracy": accuracy_score,
}

# Create the convolutional model 
cnn_module = torch.nn.Sequential( # FIXED: Squential -> nn.Sequential
    torch.nn.Conv2d(C, K, k_size, padding=k_size//2),
    torch.nn.Tanh(),
    torch.nn.Conv2d(K, 2*K, k_size, padding=k_size//2),
    torch.nn.Tanh(),
    torch.nn.Conv2d(2*K, 2*K, k_size, padding=k_size//2),
    torch.nn.Tanh(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    # FIXED: The input to Linear is (Final Channels * Height * Width). 
    # Final channels = 2*K = 32. MaxPool halves 28x28 to 14x14. 
    # So: 32 * 14 * 14 = 6272
    torch.nn.Linear(2 * K * 14 * 14, classes), 
)

# FIXED: Moved default arguments (optimizer=None, prefix=None) to the end of the parameters to prevent Python SyntaxErrors.
def run_epoch(model, data_loader, loss_func, score_funcs, results, device, optimizer=None, prefix="train"):
    
    running_loss = []
    y_true = []
    y_pred = []

    for X_batch, Y_batch in tqdm(data_loader):
        # FIXED: You must send the data batches to the same device as the model (CPU/GPU)
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        y_hat = model(X_batch)
        loss = loss_func(y_hat, Y_batch)

        if model.training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # FIXED: Fixed indentation and added () to loss.item()
        running_loss.append(loss.item()) 
         
        # FIXED: Accuracy needs actual class predictions (0-9), not raw logits. We use argmax.
        predictions = torch.argmax(y_hat, dim=1) 

        labels = Y_batch.detach().cpu().numpy()
        pred = predictions.detach().cpu().numpy()

        # FIXED: Used extend() instead of append(), and fixed typo 'labele' -> 'labels'
        # extend() prevents lists of lists, giving us one flat list of all predictions.
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist()) 

    # FIXED: We need to append the average loss to the list, not overwrite the list with a float!
    results[f"{prefix}_loss"].append(np.mean(running_loss))

    # FIXED: Added .items() to iterate through the dictionary
    for name, func in score_funcs.items():
        # FIXED: Append the score to the results list
        results[f"{prefix}_{name}_score"].append(func(y_true, y_pred)) 

def train_simple_model(model, train_loader, test_loader, loss_func, score_funcs, device, epochs=40):
    
    model.to(device) # FIXED: Moved model to device here so it happens once before training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # FIXED: Dictionary keys must be strings.
    results = {"epochs": [], "train_loss": []}
    if test_loader is not None:
        results["test_loss"] = []
    
    # FIXED: Initialize score lists for both train and test dynamically
    for name in score_funcs.keys():
        results[f"train_{name}_score"] = []
        if test_loader is not None:
            results[f"test_{name}_score"] = []

    for epoch in tqdm(range(epochs), desc="Progressing...."):
        results["epochs"].append(epoch + 1) # Track the current epoch for plotting
        
        model.train() 
        run_epoch(model, train_loader, loss_func, score_funcs, results, device, optimizer=optimizer, prefix="train")
        
        if test_loader is not None:
            model.eval()
            # FIXED: Added torch.no_grad() for the test phase to save memory (we don't need gradients here)
            with torch.no_grad():
                run_epoch(model, test_loader, loss_func, score_funcs, results, device, prefix="test")

    return results # FIXED: Return the results so we can plot them outside the function!


# FIXED: Defined the device, loss function, and actually captured the results variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_func = torch.nn.CrossEntropyLoss()

# Running for fewer epochs just to test
history = train_simple_model(cnn_module, train_loader, test_loader, loss_func, score_funcs, device, epochs=5)
