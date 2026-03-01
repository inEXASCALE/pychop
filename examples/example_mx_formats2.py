"""
Microscaling (MX) Formats Usage Examples

This script demonstrates how to use the MX formats in Pychop for:
1. Basic quantization
2. Neural network layer quantization
3. Matrix multiplication with MX
4. Compression analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pychop.mx_formats import (
    MXBlock, MXTensor, MXFloat,
    mx_quantize, compare_mx_formats,
    print_mx_format_table
)


def example_1_basic_usage():
    """Example 1: Basic MX quantization."""
    print("\n" + "="*80)
    print("Example 1: Basic MX Quantization")
    print("="*80)
    
    # Create some data
    X = np.random.randn(1024)
    
    # Quantize to MXFP8 E4M3 format
    X_mx = mx_quantize(X, format='mxfp8_e4m3', block_size=32)
    
    # Compute error
    error = np.abs(X - X_mx)
    
    print(f"Original min/max:      [{X.min():.4f}, {X.max():.4f}]")
    print(f"Quantized min/max:     [{X_mx.min():.4f}, {X_mx.max():.4f}]")
    print(f"Mean absolute error:   {error.mean():.6f}")
    print(f"Max absolute error:    {error.max():.6f}")
    print(f"Mean relative error:   {(error / (np.abs(X) + 1e-10)).mean():.4%}")


def example_2_matrix_operations():
    """Example 2: Matrix operations with MX."""
    print("\n" + "="*80)
    print("Example 2: Matrix Operations with MX")
    print("="*80)
    
    # Create weight matrix and input
    W = np.random.randn(256, 128).astype(np.float32)
    X = np.random.randn(128, 64).astype(np.float32)
    
    # Original computation
    Y_original = W @ X
    
    # MX quantized computation
    W_mx_tensor = MXTensor(W, format='mxfp8_e4m3', block_size=32)
    X_mx_tensor = MXTensor(X, format='mxfp8_e4m3', block_size=32)
    
    W_mx = W_mx_tensor.dequantize()
    X_mx = X_mx_tensor.dequantize()
    
    Y_mx = W_mx @ X_mx
    
    # Compare
    error = np.abs(Y_original - Y_mx)
    rel_error = error / (np.abs(Y_original) + 1e-10)
    
    print(f"\nWeight matrix: {W.shape}")
    print(f"Input matrix:  {X.shape}")
    print(f"Output matrix: {Y_original.shape}")
    
    print(f"\nWeight compression: {W_mx_tensor.compression_ratio():.2f}x")
    print(f"Input compression:  {X_mx_tensor.compression_ratio():.2f}x")
    
    print(f"\nOutput error:")
    print(f"  Mean absolute: {error.mean():.6f}")
    print(f"  Max absolute:  {error.max():.6f}")
    print(f"  Mean relative: {rel_error.mean():.4%}")


def example_3_format_comparison():
    """Example 3: Compare different MX formats."""
    print("\n" + "="*80)
    print("Example 3: Format Comparison")
    print("="*80)
    
    # Create test data
    X = np.random.randn(512, 256).astype(np.float32)
    
    # Compare formats
    results = compare_mx_formats(
        X,
        formats=['mxfp8_e5m2', 'mxfp8_e4m3', 'mxfp6_e3m2', 'mxfp4_e2m1'],
        block_sizes=[32]
    )
    
    # Print results
    print(f"\n{'Format':<20} {'Compression':<15} {'Storage (KB)':<15} "
          f"{'Mean Error':<15} {'Max Error':<15}")
    print("-"*85)
    
    for fmt_name, stats in sorted(results.items()):
        print(f"{fmt_name:<20} "
              f"{stats['compression_ratio_fp16']:.2f}x{'':<10} "
              f"{stats['storage_bytes']/1024:<15.2f} "
              f"{stats['mean_abs_error']:<15.6f} "
              f"{stats['max_abs_error']:<15.6f}")


def example_4_block_size_analysis():
    """Example 4: Analyze effect of block size."""
    print("\n" + "="*80)
    print("Example 4: Block Size Analysis")
    print("="*80)
    
    X = np.random.randn(2048).astype(np.float32)
    
    block_sizes = [8, 16, 32, 64, 128]
    
    print(f"\n{'Block Size':<12} {'Compression':<15} {'Mean Error':<15} {'Max Error':<15}")
    print("-"*60)
    
    for bs in block_sizes:
        mx_tensor = MXTensor(X, format='mxfp8_e4m3', block_size=bs)
        stats = mx_tensor.statistics()
        
        print(f"{bs:<12} "
              f"{stats['compression_ratio_fp16']:.3f}x{'':<10} "
              f"{stats['mean_abs_error']:<15.6f} "
              f"{stats['max_abs_error']:<15.6f}")


def example_5_neural_network_simulation():
    """Example 5: Simulate neural network with MX."""
    print("\n" + "="*80)
    print("Example 5: Neural Network Layer Simulation")
    print("="*80)
    
    # Simulate a small MLP layer
    batch_size = 128
    input_dim = 512
    hidden_dim = 256
    
    # Create layer parameters
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.random.randn(hidden_dim).astype(np.float32) * 0.01
    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    # FP32 forward pass
    Z1_fp32 = X @ W1 + b1
    A1_fp32 = np.maximum(0, Z1_fp32)  # ReLU
    
    # MX forward pass
    W1_mx = mx_quantize(W1, format='mxfp8_e4m3', block_size=32)
    X_mx = mx_quantize(X, format='mxfp8_e4m3', block_size=32)
    b1_mx = mx_quantize(b1, format='mxfp8_e4m3', block_size=32)
    
    Z1_mx = X_mx @ W1_mx + b1_mx
    A1_mx = np.maximum(0, Z1_mx)
    
    # Compare
    print(f"\nLayer: {input_dim} -> {hidden_dim}")
    print(f"Batch size: {batch_size}")
    
    print(f"\nWeight storage:")
    w_mx_tensor = MXTensor(W1, format='mxfp8_e4m3', block_size=32)
    print(f"  FP32: {W1.nbytes / 1024:.2f} KB")
    print(f"  MX:   {w_mx_tensor.storage_bits() / 8 / 1024:.2f} KB")
    print(f"  Compression: {w_mx_tensor.compression_ratio(32):.2f}x")
    
    print(f"\nActivation error (before ReLU):")
    error_z = np.abs(Z1_fp32 - Z1_mx)
    print(f"  Mean: {error_z.mean():.6f}")
    print(f"  Max:  {error_z.max():.6f}")
    
    print(f"\nActivation error (after ReLU):")
    error_a = np.abs(A1_fp32 - A1_mx)
    print(f"  Mean: {error_a.mean():.6f}")
    print(f"  Max:  {error_a.max():.6f}")


def example_6_scalar_operations():
    """Example 6: MXFloat scalar operations."""
    print("\n" + "="*80)
    print("Example 6: MXFloat Scalar Operations")
    print("="*80)
    
    # Create MXFloat numbers
    a = MXFloat(3.14159, 'mxfp8_e4m3')
    b = MXFloat(2.71828, 'mxfp8_e4m3')
    
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"\nArithmetic operations:")
    print(f"  a + b = {a + b}")
    print(f"  a - b = {a - b}")
    print(f"  a * b = {a * b}")
    print(f"  a / b = {a / b}")
    
    # Mixed operations with regular floats
    c = a + 1.0
    print(f"\nMixed operation:")
    print(f"  a + 1.0 = {c}")
    
    # Comparisons
    print(f"\nComparisons:")
    print(f"  a > b: {a > b}")
    print(f"  a < b: {a < b}")
    print(f"  a == MXFloat(3.14, 'mxfp8_e4m3'): {a == MXFloat(3.14, 'mxfp8_e4m3')}")


def example_7_compression_visualization():
    """Example 7: Visualize compression vs error tradeoff."""
    print("\n" + "="*80)
    print("Example 7: Compression vs Error Tradeoff")
    print("="*80)
    
    X = np.random.randn(1024, 512).astype(np.float32)
    
    formats = ['mxfp8_e5m2', 'mxfp8_e4m3', 'mxfp6_e3m2', 'mxfp4_e2m1']
    results = []
    
    for fmt in formats:
        mx_tensor = MXTensor(X, format=fmt, block_size=32)
        stats = mx_tensor.statistics()
        results.append({
            'format': fmt,
            'compression': stats['compression_ratio_fp16'],
            'error': stats['mean_abs_error']
        })
    
    print(f"\n{'Format':<15} {'Compression':<15} {'Mean Error':<15} {'Efficiency*':<15}")
    print("-"*65)
    print("*Efficiency = Compression / Error (higher is better)")
    print()
    
    for r in results:
        efficiency = r['compression'] / (r['error'] + 1e-10)
        print(f"{r['format']:<15} {r['compression']:<15.2f} "
              f"{r['error']:<15.6f} {efficiency:<15.1f}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Microscaling (MX) Formats - Comprehensive Examples")
    print("="*80)
    
    # Show available formats
    print_mx_format_table()
    
    # Run examples
    example_1_basic_usage()
    example_2_matrix_operations()
    example_3_format_comparison()
    example_4_block_size_analysis()
    example_5_neural_network_simulation()
    example_6_scalar_operations()
    example_7_compression_visualization()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
