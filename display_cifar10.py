import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip left/right
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), # Small rotations and shifts
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slight lighting changes
    transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)

# Loop through first 5 images
for i in range(500):
    img, label = dataset[i]  # img is a tensor [3, 32, 32]
    
    # Convert tensor to numpy array for imshow
    img_np = img.permute(1, 2, 0).numpy()  # shape [32, 32, 3]
    
    # Use seaborn style
    sns.set_style("darkgrid")
    
    plt.imshow(img_np)
    plt.title(f"CIFAR10 Sample #{i} - Label: {label}")
    plt.axis("off")
    plt.show()
    
    time.sleep(3)
