import torch.nn as nn
from pychop import ChopSTE
from pychop.layers import QuantizedLinear, QuantizedReLU
from pychop.optimizers import (
    QuantizedSGD, 
    QuantizedAdam, 
    QuantizedRMSprop, 
    QuantizedAdagrad
)

# Simple quantized model for demonstration (MLP)
class SimpleQuantizedMLP(nn.Module):
    """Simple 3-layer MLP with quantized layers for QAT demonstration."""
    def __init__(self, chop=None):
        super().__init__()
        self.fc1 = QuantizedLinear(784, 256, chop=chop)
        self.relu1 = QuantizedReLU(chop=chop)
        self.fc2 = QuantizedLinear(256, 128, chop=chop)
        self.relu2 = QuantizedReLU(chop=chop)
        self.fc3 = QuantizedLinear(128, 10, chop=chop)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten (e.g. MNIST 28x28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Define low-precision quantizers (different rounding modes)
    chop_low = ChopSTE(exp_bits=5, sig_bits=10, rmode=1)   # aggressive low precision
    chop_mid = ChopSTE(exp_bits=5, sig_bits=10, rmode=4)

    # Create model with low-precision QAT enabled
    model = SimpleQuantizedMLP(chop=chop_low)

    # Customized low-precision quantized optimizers
    optimizer_sgd = QuantizedSGD(
        model.parameters(), lr=0.01, momentum=0.9, chop=chop_low
    )

    optimizer_rmsprop = QuantizedRMSprop(
        model.parameters(), lr=0.01, chop=chop_mid
    )

    optimizer_adam = QuantizedAdam(
        model.parameters(), lr=0.001, chop=chop_low
    )

    optimizer_adagrad = QuantizedAdagrad(
        model.parameters(), lr=0.01, chop=chop_mid
    )

    print("Model and optimizers ready for low-precision QAT training.")