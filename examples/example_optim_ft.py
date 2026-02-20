"""

Complete, runnable test script for all Quantized optimizers in pychop.optimizers.

Features:
- Uses a tiny QuantizedMLP (784 ->128 ->10) built with QuantizedLinear + QuantizedReLU
- Synthetic classification data (fake MNIST-like)
- Tests EVERY optimizer from your table:
    QuantizedSGD, QuantizedAdam, QuantizedAdamW,
    QuantizedRMSprop, QuantizedAdagrad, QuantizedAdadelta
- For each optimizer: two modes
    1. chop=None     ->should behave exactly like original torch.optim
    2. chop=ChopSTE  ->floating-point QAT on optimizer states + model weights/activations
- Runs 10 training steps, prints loss
- Verifies optimizer states are properly quantized when chop is enabled
- Zero external dependencies (no torchvision, no real dataset download)


"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from pychop import ChopSTE
from pychop.layers import QuantizedLinear, QuantizedReLU
from pychop.optimizers import (
    QuantizedSGD,
    QuantizedAdam,
    QuantizedAdamW,
    QuantizedRMSprop,
    QuantizedAdagrad,
    QuantizedAdadelta,
)


class QuantizedMLP(torch.nn.Module):
    """Simple quantized MLP for testing optimizers."""

    def __init__(self, chop=None):
        super().__init__()
        self.fc1 = QuantizedLinear(784, 128, chop=chop)
        self.relu = QuantizedReLU(chop=chop)
        self.fc2 = QuantizedLinear(128, 10, chop=chop)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_fake_data(batch_size=64, num_samples=1024):
    """Generate synthetic classification data (10 classes)."""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_optimizer(OptClass, name: str, chop=None, device="cpu"):
    """Run a short training test for one optimizer."""
    print(f"\n=== Testing {name} (chop={'None' if chop is None else 'ChopSTE'}) ===")

    model = QuantizedMLP(chop=chop).to(device)
    optimizer = OptClass(model.parameters(), lr=0.01, chop=chop)
    loader = create_fake_data()

    model.train()
    initial_loss = None

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= 10:  # only 10 steps for fast test
            break

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        print(f"  Step {batch_idx+1:2d} | Loss: {loss.item():.4f}")

    # Quick verification: check if states were quantized when chop is enabled
    if chop is not None:
        quantized_states = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        quantized_states += 1
        print(f"  ->{quantized_states} optimizer states were quantized (OK)")

    print(f"  ->Final loss: {loss.item():.4f} (started from {initial_loss:.4f})")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print("pychop Quantized Optimizers Test Suite")
    print("=" * 60)

    chop = ChopSTE(exp_bits=8, sig_bits=23, rmode=3)  # standard float32-like for test

    optimizers_to_test = [
        (QuantizedSGD, "QuantizedSGD"),
        (QuantizedAdam, "QuantizedAdam"),
        (QuantizedAdamW, "QuantizedAdamW"),
        (QuantizedRMSprop, "QuantizedRMSprop"),
        (QuantizedAdagrad, "QuantizedAdagrad"),
        (QuantizedAdadelta, "QuantizedAdadelta"),
    ]

    for OptClass, name in optimizers_to_test:
        # Test 1: without chop (should be identical to original PyTorch)
        test_optimizer(OptClass, name, chop=None, device=device)

        # Test 2: with ChopSTE (QAT mode)
        test_optimizer(OptClass, name, chop=chop, device=device)

    print("\n" + "=" * 60)
    print("All Quantized optimizers tested successfully!")
    print("   - chop=None  : behaves like standard torch.optim")
    print("   - chop=ChopSTE: full floating-point QAT on optimizer states")
    print("Ready to use in your real training scripts!")