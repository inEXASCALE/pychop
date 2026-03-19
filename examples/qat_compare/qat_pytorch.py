"""
Example 3: PyChop + PyTorch (Quantized FP16)

This script demonstrates quantization-aware training (QAT) using PyChop with
PyTorch backend. It shows how to use custom floating-point precision (FP16)
for training and deployment.

MNIST digit classification
Framework: PyTorch + PyChop
Quantization: FP16 (5 exp bits, 10 sig bits) with QAT
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

import pychop
pychop.backend('torch')

from pychop import ChopSTE  # 从根目录导入 ChopSTE
from pychop.layers import (  # 从 pychop.layers 导入层
    QuantizedConv2d,
    QuantizedLinear,
    QuantizedReLU,
    QuantizedBatchNorm2d,
    QuantizedDropout,
    post_quantization
)


# ===================================================================
# 1. Define Quantized Model Architecture (PyChop + PyTorch)
# ===================================================================

class QuantizedMNISTNet(nn.Module):
    """Quantized CNN for MNIST classification using PyChop.
    
    Architecture:
    - QuantizedConv2d(1->16, 3x3) -> QuantizedReLU -> QuantizedBatchNorm -> MaxPool(2x2)
    - QuantizedConv2d(16->32, 3x3) -> QuantizedReLU -> QuantizedBatchNorm -> MaxPool(2x2)
    - Flatten
    - QuantizedLinear(32*7*7 -> 128) -> QuantizedReLU -> QuantizedDropout(0.5)
    - QuantizedLinear(128 -> 10)
    
    Attributes
    ----------
    chop : ChopSTE
        Quantizer instance with straight-through estimator.
    """
    
    def __init__(self, num_classes=10, chop=None):
        super(QuantizedMNISTNet, self).__init__()
        
        self.chop = chop
        
        # Convolutional layers with quantization
        self.conv1 = QuantizedConv2d(1, 16, kernel_size=3, padding=1, chop=chop)
        self.relu1 = QuantizedReLU(chop=chop)
        self.bn1 = QuantizedBatchNorm2d(16, chop=chop)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = QuantizedConv2d(16, 32, kernel_size=3, padding=1, chop=chop)
        self.relu2 = QuantizedReLU(chop=chop)
        self.bn2 = QuantizedBatchNorm2d(32, chop=chop)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with quantization
        self.fc1 = QuantizedLinear(32 * 7 * 7, 128, chop=chop)
        self.relu3 = QuantizedReLU(chop=chop)
        self.dropout = QuantizedDropout(p=0.5, chop=chop)
        self.fc2 = QuantizedLinear(128, num_classes, chop=chop)
    
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ===================================================================
# 2. Training Functions
# ===================================================================

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
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
    """Evaluate model on test set."""
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
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size(model, bits_per_param=32):
    """Get model size in MB."""
    num_params = count_parameters(model)
    size_mb = num_params * bits_per_param / 8 / 1024 / 1024
    return size_mb


# ===================================================================
# 3. Main Training Pipeline
# ===================================================================

def main():
    """Main QAT training pipeline with PyChop."""
    
    print("="*70)
    print("Example 3: PyChop + PyTorch (Quantized FP16 with QAT)")
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
    # Create Quantizer
    # ---------------------------------------------------------------
    print("\n[Step 2] Creating quantizer...")
    
    # FP16: 5 exponent bits, 10 significand bits (IEEE 754 half precision)
    chop = ChopSTE(exp_bits=5, sig_bits=10, rmode=1, subnormal=True)
    
    print(f"  Quantization format: FP16")
    print(f"  Exponent bits: 5")
    print(f"  Significand bits: 10")
    print(f"  Total bits: 16 (including sign bit)")
    print(f"  Unit roundoff (u): {chop.u:.6e}")
    
    # ---------------------------------------------------------------
    # Create Quantized Model for QAT
    # ---------------------------------------------------------------
    print("\n[Step 3] Creating quantized model for QAT...")
    
    model = QuantizedMNISTNet(num_classes=10, chop=chop).to(device)
    
    num_params = count_parameters(model)
    model_size_fp32 = get_model_size(model, bits_per_param=32)
    model_size_fp16 = get_model_size(model, bits_per_param=16)
    
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size (FP32): {model_size_fp32:.2f} MB")
    print(f"  Model size (FP16): {model_size_fp16:.2f} MB")
    print(f"  Size reduction: {(1 - model_size_fp16/model_size_fp32)*100:.1f}%")
    
    # ---------------------------------------------------------------
    # Training Configuration
    # ---------------------------------------------------------------
    print("\n[Step 4] Training configuration...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Epochs: {num_epochs}")
    print(f"  Training mode: Quantization-Aware Training (QAT)")
    
    # ---------------------------------------------------------------
    # Train Model with QAT
    # ---------------------------------------------------------------
    print("\n[Step 5] Training with Quantization-Aware Training...")
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
    print("\n[Step 6] Final evaluation...")
    print("="*70)
    
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    
    print(f"\nFinal Results (PyChop + PyTorch FP16 QAT):")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Model Size (deployed): {model_size_fp16:.2f} MB")
    print(f"  Precision: 16-bit (FP16)")
    print(f"  Training method: Quantization-Aware Training (QAT)")
    
    # ---------------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------------
    print("\n[Step 7] Saving quantized model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'num_parameters': num_params,
        'quantization': {
            'exp_bits': 5,
            'sig_bits': 10,
            'format': 'FP16',
            'method': 'QAT'
        }
    }, 'mnist_pychop_pytorch_fp16_qat.pth')
    
    print("  Model saved to: mnist_pychop_pytorch_fp16_qat.pth")
    
    # ---------------------------------------------------------------
    # Optional: Post-Training Quantization for Comparison
    # ---------------------------------------------------------------
    print("\n[Step 8] Bonus: PTQ from full-precision model...")
    
    # Train a full-precision model first
    print("  Training full-precision model...")
    fp_model = MNISTNet(num_classes=10).to(device)
    fp_optimizer = optim.Adam(fp_model.parameters(), lr=0.001)
    
    # Quick training (2 epochs)
    for epoch in range(1, 3):
        train_epoch(fp_model, device, train_loader, fp_optimizer, criterion, epoch)
    
    fp_loss, fp_acc = evaluate(fp_model, device, test_loader, criterion)
    print(f"  Full-precision accuracy: {fp_acc:.2f}%")
    
    chop_ptq = ChopSTE(exp_bits=5, sig_bits=10, rmode=1, subnormal=True)

    # Apply PTQ
    print("\n  Applying post-training quantization...")
    ptq_model = post_quantization(fp_model, chop_ptq, eval_mode=True, verbose=False)
    
    ptq_loss, ptq_acc = evaluate(ptq_model, device, test_loader, criterion)
    print(f"  PTQ accuracy: {ptq_acc:.2f}%")
    print(f"  Accuracy drop (PTQ): {fp_acc - ptq_acc:.2f}%")
    
    print("\n" + "="*70)
    print("Comparison: QAT vs PTQ")
    print("="*70)
    print(f"QAT (trained with quantization): {test_acc:.2f}%")
    print(f"PTQ (quantized after training):  {ptq_acc:.2f}%")
    print(f"Full Precision (baseline):       {fp_acc:.2f}%")
    print("="*70)
    
    print("\n" + "="*70)
    print("Training completed successfully! 🎉")
    print("="*70)


# Standard CNN for PTQ comparison
class MNISTNet(nn.Module):
    """Standard CNN (for PTQ comparison)."""
    
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    main()