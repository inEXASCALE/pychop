import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('../')
from pychop.optimizers import *

# Simple CNN for Testing
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Training and Evaluation Function
def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs=1, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            
        train_acc = train_correct / len(train_loader.dataset)
        
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = test_correct / len(test_loader.dataset)
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return test_acc

# Main Test Function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Test Standard vs Quantized SGD
    print("\nTraining with Standard SGD:")
    model_sgd = SimpleCNN().to(device)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
    sgd_acc = train_and_evaluate(model_sgd, optimizer_sgd, train_loader, test_loader, device=device)
    
    print("\nTraining with Quantized SGD:")
    model_qsgd = SimpleCNN().to(device)
    optimizer_qsgd = FPQuantizedSGD(model_qsgd.parameters(), lr=0.01, momentum=0.9, 
                                 ibits=8, fbits=8, rmode=1)
    qsgd_acc = train_and_evaluate(model_qsgd, optimizer_qsgd, train_loader, test_loader, device=device)

    # Test Standard vs Quantized Adam
    print("\nTraining with Standard Adam:")
    model_adam = SimpleCNN().to(device)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
    adam_acc = train_and_evaluate(model_adam, optimizer_adam, train_loader, test_loader, device=device)
    
    print("\nTraining with Quantized Adam:")
    model_qadam = SimpleCNN().to(device)
    optimizer_qadam = FPQuantizedAdam(model_qadam.parameters(), lr=0.001, 
                                   ibits=8, fbits=8, rmode=1)
    qadam_acc = train_and_evaluate(model_qadam, optimizer_qadam, train_loader, test_loader, device=device)

    # Test Standard vs Quantized RMSProp
    print("\nTraining with Standard RMSProp:")
    model_rmsprop = SimpleCNN().to(device)
    optimizer_rmsprop = torch.optim.RMSprop(model_rmsprop.parameters(), lr=0.01)
    rmsprop_acc = train_and_evaluate(model_rmsprop, optimizer_rmsprop, train_loader, test_loader, device=device)
    
    print("\nTraining with Quantized RMSProp:")
    model_qrmsprop = SimpleCNN().to(device)
    optimizer_qrmsprop = QuantizedRMSProp(model_qrmsprop.parameters(), lr=0.01, 
                                         ibits=8, fbits=8, rmode=1)
    qrmsprop_acc = train_and_evaluate(model_qrmsprop, optimizer_qrmsprop, train_loader, test_loader, device=device)

    # Test Standard vs Quantized Adagrad
    print("\nTraining with Standard Adagrad:")
    model_adagrad = SimpleCNN().to(device)
    optimizer_adagrad = torch.optim.Adagrad(model_adagrad.parameters(), lr=0.01)
    adagrad_acc = train_and_evaluate(model_adagrad, optimizer_adagrad, train_loader, test_loader, device=device)
    
    print("\nTraining with Quantized Adagrad:")
    model_qadagrad = SimpleCNN().to(device)
    optimizer_qadagrad = FPQuantizedAdagrad(model_qadagrad.parameters(), lr=0.01, 
                                         ibits=8, fbits=8, rmode=1)
    qadagrad_acc = train_and_evaluate(model_qadagrad, optimizer_qadagrad, train_loader, test_loader, device=device)

    # Test Standard vs Quantized Adadelta
    print("\nTraining with Standard Adadelta:")
    model_adadelta = SimpleCNN().to(device)
    optimizer_adadelta = torch.optim.Adadelta(model_adadelta.parameters(), lr=1.0)
    adadelta_acc = train_and_evaluate(model_adadelta, optimizer_adadelta, train_loader, test_loader, device=device)
    
    print("\nTraining with Quantized Adadelta:")
    model_qadadelta = SimpleCNN().to(device)
    optimizer_qadadelta = FPQuantizedAdadelta(model_qadadelta.parameters(), lr=1.0, 
                                           ibits=8, fbits=8, rmode=1)
    qadadelta_acc = train_and_evaluate(model_qadadelta, optimizer_qadadelta, train_loader, test_loader, device=device)

    # Test Standard vs Quantized AdamW
    print("\nTraining with Standard AdamW:")
    model_adamw = SimpleCNN().to(device)
    optimizer_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=1e-2)
    adamw_acc = train_and_evaluate(model_adamw, optimizer_adamw, train_loader, test_loader, device=device)
    
    print("\nTraining with Quantized AdamW:")
    model_qadamw = SimpleCNN().to(device)
    optimizer_qadamw = FPQuantizedAdamW(model_qadamw.parameters(), lr=0.001, weight_decay=1e-2, 
                                     ibits=8, fbits=8, rmode=1)
    qadamw_acc = train_and_evaluate(model_qadamw, optimizer_qadamw, train_loader, test_loader, device=device)

    # Final Results
    print("\nFinal Results:")
    print(f"Standard SGD Test Accuracy: {sgd_acc:.4f}")
    print(f"Quantized SGD Test Accuracy: {qsgd_acc:.4f}")
    print(f"SGD Accuracy Difference: {sgd_acc - qsgd_acc:.4f}")
    
    print(f"Standard Adam Test Accuracy: {adam_acc:.4f}")
    print(f"Quantized Adam Test Accuracy: {qadam_acc:.4f}")
    print(f"Adam Accuracy Difference: {adam_acc - qadam_acc:.4f}")
    
    print(f"Standard RMSProp Test Accuracy: {rmsprop_acc:.4f}")
    print(f"Quantized RMSProp Test Accuracy: {qrmsprop_acc:.4f}")
    print(f"RMSProp Accuracy Difference: {rmsprop_acc - qrmsprop_acc:.4f}")
    
    print(f"Standard Adagrad Test Accuracy: {adagrad_acc:.4f}")
    print(f"Quantized Adagrad Test Accuracy: {qadagrad_acc:.4f}")
    print(f"Adagrad Accuracy Difference: {adagrad_acc - qadagrad_acc:.4f}")
    
    print(f"Standard Adadelta Test Accuracy: {adadelta_acc:.4f}")
    print(f"Quantized Adadelta Test Accuracy: {qadadelta_acc:.4f}")
    print(f"Adadelta Accuracy Difference: {adadelta_acc - qadadelta_acc:.4f}")
    
    print(f"Standard AdamW Test Accuracy: {adamw_acc:.4f}")
    print(f"Quantized AdamW Test Accuracy: {qadamw_acc:.4f}")
    print(f"AdamW Accuracy Difference: {adamw_acc - qadamw_acc:.4f}")