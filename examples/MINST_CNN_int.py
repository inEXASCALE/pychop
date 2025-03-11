import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys

sys.path.append('../')

from pychop.layers import *

class QuantizedCNN(nn.Module):
    def __init__(self, num_bits=8):
        super(QuantizedCNN, self).__init__()
        self.conv1 = IntQuantizedConv2d(1, 16, 3, padding=1, num_bits=num_bits)
        self.bn1 = IntQuantizedBatchNorm2d(16, num_bits=num_bits)
        self.relu1 = IntQuantizedReLU(num_bits=num_bits)
        self.pool1 = IntQuantizedMaxPool2d(2, 2, num_bits=num_bits)
        self.conv2 = IntQuantizedConv2d(16, 32, 3, padding=1, num_bits=num_bits)
        self.bn2 = IntQuantizedBatchNorm2d(32, num_bits=num_bits)
        self.relu2 = IntQuantizedReLU(num_bits=num_bits)
        self.pool2 = IntQuantizedAvgPool2d(2, 2, num_bits=num_bits)
        self.flatten = IntQuantizedFlatten(num_bits=num_bits)
        self.fc1 = IntQuantizedLinear(32 * 7 * 7, 128, num_bits=num_bits)
        self.dropout = IntQuantizedDropout(p=0.5, num_bits=num_bits)
        self.relu3 = IntQuantizedReLU(num_bits=num_bits)
        self.fc2 = IntQuantizedLinear(128, 10, num_bits=num_bits)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

# Data Loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training Function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return total_loss / len(train_loader)

# Evaluation Function
def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy

# Main Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantizedCNN(num_bits=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
    test_acc = evaluate(model, device, test_loader)

print("\nTest Accuracy with Quantization-Aware Training:")
evaluate(model, device, test_loader)