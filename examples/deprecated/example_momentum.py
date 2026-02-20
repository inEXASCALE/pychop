# Test the optimizers

import torch 
import sys
sys.path.append('../')
from pychop.optimizers import *
from pychop.layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
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

    quant = QuantizedLayer(exp_bits=5, sig_bits=10, rmode=1) 
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
            if param.grad is not None:
                param.grad.data = quant(param.grad.data) # Quantize gradient accumulators
            print(f"{param_name} - Updated: {param.data.norm():.4f}, Grad exists: {param.grad is not None}")
            
        print(f"Loss: {loss.item():.4f}")

