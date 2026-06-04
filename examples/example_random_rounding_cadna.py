"""
Example usage of CADNA-style random rounding (rmode=7).
"""

import numpy as np
import pychop
from pychop import Chop

# ============================================================
# Example 1: NumPy Backend
# ============================================================
print("=" * 60)
print("NumPy Backend - CADNA Rounding (rmode=7)")
print("=" * 60)

pychop.backend('numpy')

# Create chopper with CADNA rounding
ch_cadna = Chop(exp_bits=5, sig_bits=10, rmode=7, random_state=42)
ch_standard = Chop(exp_bits=5, sig_bits=10, rmode=5, random_state=42)

# Test data
x = np.array([1.23456789, -2.34567890, 3.45678901, -4.56789012])

print("\nOriginal values:")
print(x)

print("\nCADNA rounding (10 runs):")
for i in range(10):
    result = ch_cadna(x.copy())
    print(f"Run {i+1}: {result}")

print("\nStandard stochastic rounding (10 runs):")
for i in range(10):
    result = ch_standard(x.copy())
    print(f"Run {i+1}: {result}")


# ============================================================
# Example 2: PyTorch Backend (if available)
# ============================================================
try:
    import torch
    
    print("\n" + "=" * 60)
    print("PyTorch Backend - CADNA Rounding (rmode=7)")
    print("=" * 60)
    
    pychop.backend('torch')
    
    ch = Chop(exp_bits=5, sig_bits=10, rmode=7, random_state=42)
    
    x_torch = torch.tensor([1.23456789, -2.34567890, 3.45678901, -4.56789012])
    
    print("\nOriginal tensor:")
    print(x_torch)
    
    print("\nCADNA rounding (5 runs):")
    for i in range(5):
        result = ch(x_torch.clone())
        print(f"Run {i+1}: {result}")

except ImportError:
    print("\n⊘ PyTorch not available, skipping PyTorch example")


# ============================================================
# Example 3: Performance Comparison
# ============================================================
print("\n" + "=" * 60)
print("Performance Comparison")
print("=" * 60)

import time

pychop.backend('numpy')

# Large array
x_large = np.random.randn(10000)

# Test different rounding modes
modes = {
    'rmode=5 (proportional)': Chop(exp_bits=8, sig_bits=23, rmode=5),
    'rmode=6 (uniform)': Chop(exp_bits=8, sig_bits=23, rmode=6),
    'rmode=7 (CADNA)': Chop(exp_bits=8, sig_bits=23, rmode=7)
}

num_runs = 100

for name, ch in modes.items():
    start = time.time()
    for _ in range(num_runs):
        _ = ch(x_large.copy())
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed:.4f} seconds ({num_runs} runs)")


# ============================================================
# Example 4: Statistical Properties
# ============================================================
print("\n" + "=" * 60)
print("Statistical Properties")
print("=" * 60)

pychop.backend('numpy')

x_test = np.array([1.5])  # Exact midpoint
ch = Chop(exp_bits=8, sig_bits=23, rmode=7, random_state=42)

# Run many times
results = np.array([ch(x_test.copy())[0] for _ in range(1000)])

print(f"\nOriginal value: {x_test[0]}")
print(f"Mean of rounded values: {results.mean():.6f}")
print(f"Std of rounded values: {results.std():.6f}")
print(f"Min: {results.min():.6f}, Max: {results.max():.6f}")
print(f"Unique values: {np.unique(results)}")


print("\n✅ All examples completed!")