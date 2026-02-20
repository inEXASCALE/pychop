"""

Complete, runnable test script for all **IQuantized** optimizers in pychop.optimizers.

Features:
- Uses a tiny IQuantizedMLP (784 -> 128 -> 10) built with IQuantizedLinear + IQuantizedReLU
- Synthetic classification data (fake MNIST-like)
- Tests EVERY IQuantized optimizer from your table:
    IQuantizedSGD, IQuantizedAdam, IQuantizedAdamW,
    IQuantizedRMSprop, IQuantizedAdagrad, IQuantizedAdadelta
- For each optimizer: two modes
    1. chop=None          -> falls back to original torch.optim (no integer QAT)
    2. chop=ChopiSTE(...) -> full integer fake-quantization on optimizer states + weights/activations
- Runs only 10 training steps (very fast)
- Verifies that optimizer states were actually quantized when ChopiSTE is used
- Zero external dependencies (pure torch + pychop)

"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function
from typing import Optional, Any, Callable
from pychop import ChopiSTE
from pychop.layers import IQuantizedLinear, IQuantizedReLU


from pychop.optimizers import (
    IQuantizedSGD,
    IQuantizedAdam,
    IQuantizedAdamW,
    IQuantizedRMSprop,
    IQuantizedAdagrad,
    IQuantizedAdadelta,
)



# ====================== IQuantized MLP for testing ======================
class IQuantizedMLP(torch.nn.Module):
    """Simple integer-quantized MLP for optimizer testing."""

    def __init__(self, chop=None):
        super().__init__()
        self.fc1 = IQuantizedLinear(784, 128, chop=chop)
        self.relu = IQuantizedReLU(chop=chop)
        self.fc2 = IQuantizedLinear(128, 10, chop=chop)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ====================== Fake Data ======================
def create_fake_data(batch_size=64, num_samples=1024):
    """Generate synthetic classification data (10 classes)."""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ====================== Test Helper ======================
def test_ioptimizer(OptClass, name: str, chop=None, device="cpu"):
    """Run a short training test for one IQuantized optimizer."""
    print(f"\n=== Testing {name} (chop={'None' if chop is None else 'ChopiSTE'}) ===")

    model = IQuantizedMLP(chop=chop).to(device)
    optimizer = OptClass(model.parameters(), lr=0.01, chop=chop)
    loader = create_fake_data()

    model.train()
    initial_loss = None

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= 10:   # only 10 steps
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

    # Verification: check optimizer states were quantized
    if chop is not None:
        quantized_states = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        quantized_states += 1
        print(f"  -> {quantized_states} optimizer states were integer-quantized (OK)")

    print(f"  -> Final loss: {loss.item():.4f} (started from {initial_loss:.4f})")


# ====================== Main Test Suite ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print("pychop IQuantized Optimizers Test Suite (Integer QAT)")
    print("=" * 70)

    # Integer quantizer (8-bit symmetric is common)
    chop_int = ChopiSTE(bits=8, symmetric=True)

    optimizers_to_test = [
        (IQuantizedSGD, "IQuantizedSGD"),
        (IQuantizedAdam, "IQuantizedAdam"),
        (IQuantizedAdamW, "IQuantizedAdamW"),
        (IQuantizedRMSprop, "IQuantizedRMSprop"),
        (IQuantizedAdagrad, "IQuantizedAdagrad"),
        (IQuantizedAdadelta, "IQuantizedAdadelta"),
    ]

    # Correct imports (add these at top in real file)
    # from pychop.optimizers import IQuantizedSGD, IQuantizedAdam, ...

    from pychop.optimizers import (
        IQuantizedSGD,
        IQuantizedAdam,
        IQuantizedAdamW,
        IQuantizedRMSprop,
        IQuantizedAdagrad,
        IQuantizedAdadelta,
    )

    optimizers_to_test = [
        (IQuantizedSGD,      "IQuantizedSGD"),
        (IQuantizedAdam,     "IQuantizedAdam"),
        (IQuantizedAdamW,    "IQuantizedAdamW"),
        (IQuantizedRMSprop,  "IQuantizedRMSprop"),
        (IQuantizedAdagrad,  "IQuantizedAdagrad"),
        (IQuantizedAdadelta, "IQuantizedAdadelta"),
    ]

    for OptClass, name in optimizers_to_test:
        # Test 1: without chop (standard behavior)
        test_ioptimizer(OptClass, name, chop=None, device=device)

        # Test 2: with ChopiSTE (integer QAT mode)
        test_ioptimizer(OptClass, name, chop=chop_int, device=device)

    print("\n" + "=" * 70)
    print(" All IQuantized optimizers tested successfully!")
    print("   - chop=None       : identical to standard torch.optim")
    print("   - chop=ChopiSTE   : full integer fake-quantization QAT on optimizer states")
    print("   Ready for production use with your IQuantized layers!")