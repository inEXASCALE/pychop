import torch
import torch.nn as nn

import sys
# appending a path
sys.path.append('../')

from pychop.layers import QuantizedLinear, QuantizedBatchNorm2d, QuantizedLayerNorm, QuantizedMultiheadAttention, QuantizedGRU
from pychop.optimizers import *

def test_quantized_optimizers():
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = QuantizedLinear(10, 5, exp_bits=5, sig_bits=10, rmode=1)
        
        def forward(self, x):
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestNet().to(device)
    x = torch.randn(2, 10, device=device, requires_grad=True)
    target = torch.randn(2, 5, device=device)
    criterion = nn.MSELoss()

    # Test each optimizer
    optimizers = [
        ("SGD", QuantizedSGD(model.parameters(), lr=0.01, momentum=0.9, exp_bits=5, sig_bits=10, rmode=1)),
        ("RMSprop", QuantizedRMSprop(model.parameters(), lr=0.01, exp_bits=5, sig_bits=10, rmode=5)),
        ("Adagrad", QuantizedAdagrad(model.parameters(), lr=0.01, exp_bits=5, sig_bits=10, rmode=4)),
        ("Adam", QuantizedAdam(model.parameters(), lr=0.001, exp_bits=5, sig_bits=10, rmode=6))
    ]

    for name, optimizer in optimizers:
        print(f"\nTesting {name}:")
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        for param_name, param in model.named_parameters():
            print(f"{param_name} - Updated: {param.data.norm():.4f}, Grad exists: {param.grad is not None}")
        print(f"Loss: {loss.item():.4f}")

# Test the optimizer
def test_quantized_adam_optimizers():
    # Simple model
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = QuantizedLinear(10, 5, exp_bits=5, sig_bits=10, rmode=1)
        
        def forward(self, x):
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestNet().to(device)
    
    # Optimizer with quantized momentum and accumulators
    optimizer = QuantizedAdam(model.parameters(), lr=0.01, exp_bits=5, sig_bits=10, rmode=6)
    
    # Dummy input and target
    x = torch.randn(2, 10, device=device, requires_grad=True)
    target = torch.randn(2, 5, device=device)
    
    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    
    # Check if parameters updated and gradients exist
    for name, param in model.named_parameters():
        print(f"{name} - Updated: {param.data.norm()}, Grad exists: {param.grad is not None}")
    print(f"Loss: {loss.item()}")

# Test the layers
def test_layers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test QuantizedBatchNorm2d
    bn_layer = QuantizedBatchNorm2d(3, 5, 10, rmode=1).to(device)
    x_bn = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    y_bn = bn_layer(x_bn)
    loss_bn = y_bn.sum()
    loss_bn.backward()
    print("Quantized BatchNorm2d - Output shape:", y_bn.shape, "Gradients:", x_bn.grad is not None)
    
    # Test QuantizedLayerNorm
    ln_layer = QuantizedLayerNorm(512, 5, 10, rmode=4).to(device)
    x_ln = torch.randn(2, 10, 512, device=device, requires_grad=True)
    y_ln = ln_layer(x_ln)
    loss_ln = y_ln.sum()
    loss_ln.backward()
    print("Quantized LayerNorm - Output shape:", y_ln.shape, "Gradients:", x_ln.grad is not None)
    
    # Test QuantizedGRU
    gru_layer = QuantizedGRU(10, 20, 5, 10, num_layers=1, rmode=5).to(device)
    x_gru = torch.randn(2, 5, 10, device=device, requires_grad=True)
    y_gru, h_gru = gru_layer(x_gru)
    loss_gru = y_gru.sum()
    loss_gru.backward()
    print("Quantized GRU - Output shape:", y_gru.shape, "Gradients:", x_gru.grad is not None)
    
    # Test QuantizedMultiheadAttention
    mha_layer = QuantizedMultiheadAttention(512, 8, 5, 10, rmode=1).to(device)
    x_mha = torch.randn(2, 10, 512, device=device, requires_grad=True)
    y_mha, attn_weights = mha_layer(x_mha, x_mha, x_mha)
    loss_mha = y_mha.sum()
    loss_mha.backward()
    print("Quantized MultiheadAttention - Output shape:", y_mha.shape, "Gradients:", x_mha.grad is not None)

if __name__ == "__main__":
    test_layers()
    test_quantized_optimizers()
    test_quantized_adam_optimizers()