"""
Example 1: Raw PyTorch Baseline (FP32)

This script demonstrates standard training and deployment of a CNN classifier
using raw PyTorch without any quantization. This serves as the baseline for
comparing quantized models.

MNIST digit classification
Framework: PyTorch (FP32)
Quantization: None
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np


# ===================================================================
# 1. Define Model Architecture (Raw PyTorch)
# ===================================================================

class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification.
    
    Architecture:
    - Conv2d(1->16, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)
    - Conv2d(16->32, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)
    - Flatten
    - Linear(32*7*7 -> 128) -> ReLU -> Dropout(0.5)
    - Linear(128 -> 10)
    """
    
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ===================================================================
# 2. Training Functions
# ===================================================================

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    device : torch.device
        Device to use.
    train_loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    criterion : nn.Module
        Loss function.
    epoch : int
        Current epoch number.
    
    Returns
    -------
    tuple
        (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={epoch_time:.2f}s")
    
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """Evaluate model on test set.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    device : torch.device
        Device to use.
    test_loader : DataLoader
        Test data loader.
    criterion : nn.Module
        Loss function.
    
    Returns
    -------
    tuple
        (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def count_parameters(model):
    """Count total number of parameters.
    
    Parameters
    ----------
    model : nn.Module
        Model to count parameters for.
    
    Returns
    -------
    int
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def get_model_size(model):
    """Get model size in MB.
    
    Parameters
    ----------
    model : nn.Module
        Model to measure.
    
    Returns
    -------
    float
        Model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# ===================================================================
# 3. Main Training Pipeline
# ===================================================================

def main():
    """Main training and evaluation pipeline."""
    
    print("="*70)
    print("Example 1: Raw PyTorch Baseline (FP32)")
    print("="*70)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ---------------------------------------------------------------
    # Load MNIST Dataset
    # ---------------------------------------------------------------
    print("\n[Step 1] Loading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # ---------------------------------------------------------------
    # Create Model
    # ---------------------------------------------------------------
    print("\n[Step 2] Creating model...")
    
    model = MNISTNet(num_classes=10).to(device)
    
    num_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: {model_size:.2f} MB (FP32)")
    print(f"  Precision: FP32 (32-bit floating point)")
    
    # ---------------------------------------------------------------
    # Training Configuration
    # ---------------------------------------------------------------
    print("\n[Step 3] Training configuration...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Epochs: {num_epochs}")
    
    # ---------------------------------------------------------------
    # Train Model
    # ---------------------------------------------------------------
    print("\n[Step 4] Training model...")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        print(f"  Test: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print("-"*70)
    
    # ---------------------------------------------------------------
    # Final Evaluation
    # ---------------------------------------------------------------
    print("\n[Step 5] Final evaluation...")
    print("="*70)
    
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    
    print(f"\nFinal Results (Raw PyTorch FP32):")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Model Size: {model_size:.2f} MB")
    print(f"  Precision: 32-bit")
    
    # ---------------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------------
    print("\n[Step 6] Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'num_parameters': num_params,
    }, 'mnist_pytorch_fp32.pth')
    
    print("  Model saved to: mnist_pytorch_fp32.pth")
    
    print("\n" + "="*70)
    print("Training completed successfully! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()