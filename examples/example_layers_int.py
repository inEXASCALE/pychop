import sys
# appending a path
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pychop.layers import *


class TestDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=10, input_dim=5, num_classes=2):
        self.data = torch.randn(num_samples, seq_len, input_dim)  # [batch, seq_len, input_dim]
        self.labels = torch.randint(0, num_classes, (num_samples,))  # Classification labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class TestQuantizedModel(nn.Module):
    def __init__(self, num_bits=8):
        super(TestQuantizedModel, self).__init__()
        # 1D Convolution for sequence data
        self.conv1d = IntQuantizedConv1d(5, 8, kernel_size=3, padding=1, num_bits=num_bits)
        # LSTM for sequential processing
        self.lstm = IntQuantizedLSTM(8, 16, num_layers=1, num_bits=num_bits)
        # GRU for additional recurrent processing
        self.gru = IntQuantizedGRU(16, 16, num_layers=1, num_bits=num_bits)
        # Attention mechanism
        self.attn = IntQuantizedAttention(embed_dim=16, num_heads=2, num_bits=num_bits)
        # Linear layers
        self.fc1 = IntQuantizedLinear(16, 32, num_bits=num_bits)
        self.relu = IntQuantizedReLU(num_bits=num_bits)
        self.fc2 = IntQuantizedLinear(32, 2, num_bits=num_bits)  # 2 classes

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len] for Conv1d
        x = self.conv1d(x)     # [batch, 8, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, 8] for LSTM/GRU
        x, _ = self.lstm(x)    # [batch, seq_len, 16]
        x, _ = self.gru(x)     # [batch, seq_len, 16]
        x = self.attn(x, x, x) # [batch, seq_len, 16]
        x = x[:, -1, :]        # Take last timestep: [batch, 16]
        x = self.fc1(x)        # [batch, 32]
        x = self.relu(x)
        x = self.fc2(x)        # [batch, 2]
        return x

# Training and Evaluation Functions
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
    print(f'Train Epoch: {epoch}, Loss: {total_loss / len(train_loader):.6f}')
    return total_loss / len(train_loader)

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TestDataset(num_samples=100, seq_len=10, input_dim=5, num_classes=2)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)  

model = TestQuantizedModel(num_bits=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
    test_acc = evaluate(model, device, test_loader)

print("\nTest Accuracy with Quantization-Aware Training:")
evaluate(model, device, test_loader)