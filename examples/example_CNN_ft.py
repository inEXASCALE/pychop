# old version, deprecated
import sys
# appending a path
sys.path.append('../')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pychop.layers import *
from pychop import LightChopSTE

class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        return self.fc2(x)

# Quantized CNN
class QuantizedCNN(nn.Module):
    def __init__(self, chop):
        super().__init__()
        self.conv1 = QuantizedConv2d(1, 16, 3, chop=chop)
        self.pool = QuantizedMaxPool2d(2, chop=chop)
        self.conv2 = QuantizedConv2d(16, 32, 3, chop=chop)
        # self.pool2 = QuantizedMaxPool2d(2, chop)
        self.fc1 = QuantizedLinear(32 * 5 * 5, 128, chop=chop)
        # self.dropout = QuantizedDropout(0.5, chop)
        self.fc2 = QuantizedLinear(128, 10, chop=chop)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

# Test functions
def train_and_evaluate(model, train_loader, test_loader, epochs=2, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
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

def test_layers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chop=LightChopSTE(exp_bits=5, sig_bits=10, rmode=3) # half precision
    # Test Conv1d
    conv1d = nn.Conv1d(3, 6, 3).to(device)
    qconv1d = QuantizedConv1d(3, 6, 3, chop=chop).to(device)
    x = torch.randn(1, 3, 32, device=device)
    diff = torch.norm(conv1d(x) - qconv1d(x))
    print(f"Conv1d difference: {diff.item():.4f}")

    # Test Conv3d
    conv3d = nn.Conv3d(3, 6, 3).to(device)
    qconv3d = QuantizedConv3d(3, 6, 3, chop=chop).to(device)
    x = torch.randn(1, 3, 16, 16, 16, device=device)
    diff = torch.norm(conv3d(x) - qconv3d(x))
    print(f"Conv3d difference: {diff.item():.4f}")

    # Test MaxPool1d
    maxpool1d = nn.MaxPool1d(2).to(device)
    qmaxpool1d = QuantizedMaxPool1d(2, chop=chop).to(device)
    x = torch.randn(1, 3, 32, device=device)
    diff = torch.norm(maxpool1d(x) - qmaxpool1d(x))
    print(f"MaxPool1d difference: {diff.item():.4f}")

    # Test MaxPool2d
    maxpool2d = nn.MaxPool2d(2).to(device)
    qmaxpool2d = QuantizedMaxPool2d(2, chop=chop).to(device)
    x = torch.randn(1, 3, 32, 32, device=device)
    diff = torch.norm(maxpool2d(x) - qmaxpool2d(x))
    print(f"MaxPool2d difference: {diff.item():.4f}")

    # Test MaxPool3d
    maxpool3d = nn.MaxPool3d(2).to(device)
    qmaxpool3d = QuantizedMaxPool3d(2, chop=chop).to(device)
    x = torch.randn(1, 3, 16, 16, 16, device=device)
    diff = torch.norm(maxpool3d(x) - qmaxpool3d(x))
    print(f"MaxPool3d difference: {diff.item():.4f}")

    # Test AvgPool
    avgpool = nn.AvgPool2d(2).to(device)
    qavgpool = QuantizedAvgPool(2, chop=chop).to(device)
    x = torch.randn(1, 3, 32, 32, device=device)
    diff = torch.norm(avgpool(x) - qavgpool(x))
    print(f"AvgPool difference: {diff.item():.4f}")

    # Test AvgPool1d
    avgpool1d = nn.AvgPool1d(2).to(device)
    qavgpool1d = QuantizedAvgPool1d(2, chop=chop).to(device)
    x = torch.randn(1, 3, 32, device=device)
    diff = torch.norm(avgpool1d(x) - qavgpool1d(x))
    print(f"AvgPool1d difference: {diff.item():.4f}")

    # Test AvgPool2d
    avgpool2d = nn.AvgPool2d(2).to(device)
    qavgpool2d = QuantizedAvgPool2d(2, chop=chop).to(device)
    x = torch.randn(1, 3, 32, 32, device=device)
    diff = torch.norm(avgpool2d(x) - qavgpool2d(x))
    print(f"AvgPool2d difference: {diff.item():.4f}")

    # Test LSTM
    lstm = nn.LSTM(10, 20, batch_first=True).to(device)
    qlstm = QuantizedLSTM(10, 20, chop=chop).to(device)
    x = torch.randn(2, 5, 10, device=device)
    output, _ = lstm(x)
    qoutput, _ = qlstm(x)
    diff = torch.norm(output - qoutput)
    print(f"LSTM difference: {diff.item():.4f}")

    # Test Attention
    embed_dim = 512
    num_heads = 8
    
    # 1. Ensure both have exactly the same embedding dimension and number of heads
    attn = nn.MultiheadAttention(embed_dim, num_heads).to(device)
    qattn = QuantizedAttention(embed_dim, num_heads, chop=chop).to(device)
    
    # 2. Copy weights: to compare quantization error, qattn must have exactly the same initial weights as attn
    qattn.load_state_dict(attn.state_dict())
    
    # Note: By default batch_first=False, so the input shape should be (Sequence Length, Batch Size, Embedding Dimension)
    x = torch.randn(10, 2, 512, device=device)
    
    # 3. Pass in Query, Key, Value (in self-attention they are all x)
    output, _ = attn(x, x, x)
    qoutput, _ = qattn(x, x, x)  # Fixed the call here
    
    diff = torch.norm(output - qoutput)
    print(f"Attention difference: {diff.item():.4f}")

    # Test Dropout
    dropout = nn.Dropout(0.5).to(device)
    qdropout = QuantizedDropout(0.5, chop=chop).to(device)
    x = torch.randn(2, 512, device=device)
    diff = torch.norm(dropout(x) - qdropout(x))
    print(f"Dropout difference: {diff.item():.4f}")

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

    # Test individual layers
    print("Testing individual layers:")
    test_layers()
    
    print("\nTraining Quantized CNN:")
    chop=LightChopSTE(exp_bits=5, sig_bits=10, rmode=3) # half precision
    quantized_model = QuantizedCNN(chop=chop)
    quantized_acc = train_and_evaluate(quantized_model, train_loader, test_loader, device=device)

    # Test full models
    print("\nTraining Standard CNN:")
    standard_model = StandardCNN()
    standard_acc = train_and_evaluate(standard_model, train_loader, test_loader, device=device)
    
    print(f"\nFinal Results:")
    print(f"Quantized CNN Test Accuracy: {quantized_acc:.4f}")
    print(f"Accuracy Difference: {standard_acc - quantized_acc:.4f}")