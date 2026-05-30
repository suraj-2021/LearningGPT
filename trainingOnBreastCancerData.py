import time
import numpy as np
import pandas as pd
import torch 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm 

# ==========================================
# 1. LOAD AND PREPARE REAL-WORLD DATA
# ==========================================
print("Loading Breast Cancer dataset...")
# Load the data directly from scikit-learn
cancer_data = load_breast_cancer()
X = cancer_data.data   # The 30 features
y = cancer_data.target # The labels (0 = Malignant, 1 = Benign)

print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features.")

# Split into 80% training and 20% testing BEFORE scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (CRITICAL for real-world data with mixed ranges)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Only transform the test set based on train set's scale

# Convert to PyTorch TensorDatasets
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)) 
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)) 

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32) 

# ==========================================
# 2. BUILD THE NEURAL NETWORK ARCHITECTURE
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
score_funcs = {'Accuracy': accuracy_score, 'F1': f1_score}

# Notice the input is now 30, matching our feature count!
model = torch.nn.Sequential(
       torch.nn.Linear(in_features=30, out_features=64),
       torch.nn.ReLU(), # Switched to ReLU, often better for real-world data than Tanh
       torch.nn.Linear(in_features=64, out_features=32),
       torch.nn.ReLU(),
       torch.nn.Linear(in_features=32, out_features=2) # 2 outputs for binary classification
)

# ==========================================
# 3. THE TRAINING ENGINE (run_epoch)
# ==========================================
def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time() 
    
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_hat = model(X_batch)
        loss = loss_func(y_hat, y_batch)
        
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        running_loss.append(loss.item())
        
        if len(score_funcs) > 0:
            labels = y_batch.detach().cpu().numpy()
            predictions = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
            
    end = time.time()
    results[f"{prefix} loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        results[f"{prefix} {name}"].append(score_func(y_true, y_pred))
        
    return end - start 

# ==========================================
# 4. THE MANAGER FUNCTION
# ==========================================
def train_simple_network(model, loss_func, train_loader, test_loader=None, score_funcs=None, epochs=50, device="cpu"):
    
    # We are using Adam here. It is generally much faster and more reliable than SGD for real data.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    
    results = {"epoch": [], "total time": []}
    prefixes = ["train"]
    if test_loader is not None:
        prefixes.append("test")
        
    for prefix in prefixes:
        results[f"{prefix} loss"] = []
        if score_funcs is not None:
            for name in score_funcs.keys():
                results[f"{prefix} {name}"] = []

    total_train_time = 0.0
    for epoch in tqdm(range(epochs), desc="Training"):
        
        model = model.train() 
        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train")
        
        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)
        
        if test_loader is not None:
            model = model.eval() 
            with torch.no_grad(): 
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test")
                
    return pd.DataFrame.from_dict(results)


loss_function = torch.nn.CrossEntropyLoss() 

print("\nStarting Training...")
df_results = train_simple_network(
    model=model, 
    loss_func=loss_function, 
    train_loader=train_loader, 
    test_loader=test_loader, 
    score_funcs=score_funcs, 
    epochs=40, # Usually needs fewer epochs with Adam optimizer
    device=device
)

print("\nFinal Results on Real-World Data (Last 5 Epochs):")
print(df_results[['epoch', 'train Accuracy', 'test Accuracy', 'train F1', 'test F1']].tail())
