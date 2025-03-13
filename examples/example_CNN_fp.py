
import sys
# appending a path
sys.path.append('../')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pychop.layers import *


# CNN Models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)   
        self.fc2 = nn.Linear(128, 10)          

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QuantizedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FPQuantizedConv2d(1, 16, 3, 1, 1, int_bits=8, frac_bits=8)
        self.pool = FPQuantizedMaxPool2d(2, 2, int_bits=8, frac_bits=8)
        self.conv2 = FPQuantizedConv2d(16, 32, 3, 1, 1, int_bits=8, frac_bits=8)
        self.fc1 = FPQuantizedLinear(32 * 7 * 7, 128, int_bits=8, frac_bits=8)
        self.fc2 = FPQuantizedLinear(128, 10, int_bits=8, frac_bits=8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training and Evaluation Function
def train_and_evaluate(model, train_loader, test_loader, epochs=5, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy, avg_test_loss

if __name__ == '__main__':
    # Load MNIST Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Train and Compare
    print("Training Normal CNN:")
    normal_cnn = CNN()
    normal_acc, normal_loss = train_and_evaluate(normal_cnn, train_loader, test_loader, epochs=5, device=device)

    print("\nTraining Quantized CNN:")
    quantized_cnn = QuantizedCNN()
    quantized_acc, quantized_loss = train_and_evaluate(quantized_cnn, train_loader, test_loader, epochs=5, device=device)

    # Summary
    print("\nComparison Summary:")
    print(f"Normal CNN - Test Loss: {normal_loss:.4f}, Accuracy: {normal_acc:.2f}%")
    print(f"Quantized CNN - Test Loss: {quantized_loss:.4f}, Accuracy: {quantized_acc:.2f}%")
    print(f"Accuracy Difference (Normal - Quantized): {normal_acc - quantized_acc:.2f}%")


