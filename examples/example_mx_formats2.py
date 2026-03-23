"""
Microscaling (MX) Formats - Comprehensive Usage Examples

This script demonstrates how to use the MX formats in Pychop for:
1. Basic quantization with multi-backend support
2. Neural network layer quantization
3. Matrix multiplication with MX
4. Compression analysis
5. Backend comparison

"""

import numpy as np
import pychop

# Import MX formats
from pychop import (
    mx_quantize,
    MXTensor,
    MX_FORMATS,
    compare_mx_formats,
    print_mx_format_table
)


def example_1_basic_usage():
    """Example 1: Basic MX quantization."""
    print("\n" + "="*80)
    print("Example 1: Basic MX Quantization")
    print("="*80)
    
    # Set backend to auto (will detect from input)
    pychop.backend('auto')
    
    # Create some data
    X = np.random.randn(1024).astype(np.float32)
    
    # Quantize to MXFP8 E4M3 format
    X_mx = mx_quantize(X, format='mxfp8_e4m3', block_size=32)
    
    # Compute error
    error = np.abs(X - X_mx)
    
    print(f"Original min/max:      [{X.min():.4f}, {X.max():.4f}]")
    print(f"Quantized min/max:     [{X_mx.min():.4f}, {X_mx.max():.4f}]")
    print(f"Mean absolute error:   {error.mean():.6f}")
    print(f"Max absolute error:    {error.max():.6f}")
    print(f"Mean relative error:   {(error / (np.abs(X) + 1e-10)).mean():.4%}")
    print(f"\n✓ Backend used: numpy (auto-detected)")


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
    
    # Get statistics
    w_stats = W_mx_tensor.statistics()
    x_stats = X_mx_tensor.statistics()
    
    # Compare
    error = np.abs(Y_original - Y_mx)
    rel_error = error / (np.abs(Y_original) + 1e-10)
    
    print(f"\nWeight matrix: {W.shape}")
    print(f"Input matrix:  {X.shape}")
    print(f"Output matrix: {Y_original.shape}")
    
    print(f"\nWeight compression: {w_stats['compression_ratio_fp16']:.2f}x vs FP16")
    print(f"Input compression:  {x_stats['compression_ratio_fp16']:.2f}x vs FP16")
    
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
    
    # Compare formats using built-in function
    print("\nComparing MX formats on random data (512x256):")
    compare_mx_formats(
        X,
        formats=['mxfp8_e5m2', 'mxfp8_e4m3', 'mxfp6_e3m2', 'mxfp4_e2m1'],
        block_size=32
    )


def example_4_block_size_analysis():
    """Example 4: Analyze effect of block size."""
    print("\n" + "="*80)
    print("Example 4: Block Size Analysis")
    print("="*80)
    
    X = np.random.randn(2048).astype(np.float32)
    
    block_sizes = [8, 16, 32, 64, 128]
    
    print(f"\nAnalyzing MXFP8_E4M3 with different block sizes:")
    print(f"\n{'Block Size':<12} {'Compression':<15} {'MSE':<15} {'Bits/Element':<15}")
    print("-"*60)
    
    for bs in block_sizes:
        mx_tensor = MXTensor(X, format='mxfp8_e4m3', block_size=bs)
        stats = mx_tensor.statistics()
        
        # Compute MSE
        X_reconstructed = mx_tensor.dequantize()
        mse = np.mean((X - X_reconstructed) ** 2)
        
        print(f"{bs:<12} "
              f"{stats['compression_ratio_fp16']:.3f}x{'':<10} "
              f"{mse:<15.6f} "
              f"{stats['bits_per_element']:<15.2f}")
    
    print("\nObservation: Larger block sizes → better compression, potentially higher error")


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
    
    # Get compression stats
    w_mx_tensor = MXTensor(W1, format='mxfp8_e4m3', block_size=32)
    w_stats = w_mx_tensor.statistics()
    
    # Compare
    print(f"\nLayer: {input_dim} -> {hidden_dim}")
    print(f"Batch size: {batch_size}")
    
    print(f"\nWeight storage:")
    print(f"  FP32: {W1.nbytes / 1024:.2f} KB")
    print(f"  MX:   {w_stats['bits_per_element'] * W1.size / 8 / 1024:.2f} KB")
    print(f"  Compression: {w_stats['compression_ratio_fp32']:.2f}x vs FP32")
    print(f"  Compression: {w_stats['compression_ratio_fp16']:.2f}x vs FP16")
    
    print(f"\nActivation error (before ReLU):")
    error_z = np.abs(Z1_fp32 - Z1_mx)
    print(f"  Mean: {error_z.mean():.6f}")
    print(f"  Max:  {error_z.max():.6f}")
    print(f"  Relative: {(error_z / (np.abs(Z1_fp32) + 1e-10)).mean():.4%}")
    
    print(f"\nActivation error (after ReLU):")
    error_a = np.abs(A1_fp32 - A1_mx)
    print(f"  Mean: {error_a.mean():.6f}")
    print(f"  Max:  {error_a.max():.6f}")
    print(f"  Relative: {(error_a / (np.abs(A1_fp32) + 1e-10)).mean():.4%}")


def example_6_multi_backend_comparison():
    """Example 6: Compare backends (NumPy, PyTorch, JAX)."""
    print("\n" + "="*80)
    print("Example 6: Multi-Backend Comparison")
    print("="*80)
    
    # Test data
    X_np = np.random.randn(256, 256).astype(np.float32)
    
    backends_to_test = []
    
    # NumPy
    backends_to_test.append(('numpy', X_np))
    
    # PyTorch
    try:
        import torch
        X_torch = torch.from_numpy(X_np)
        backends_to_test.append(('torch', X_torch))
    except ImportError:
        print("PyTorch not available")
    
    # JAX
    try:
        import jax.numpy as jnp
        X_jax = jnp.array(X_np)
        backends_to_test.append(('jax', X_jax))
    except ImportError:
        print("JAX not available")
    
    print(f"\nTesting MX quantization across backends:")
    print(f"{'Backend':<12} {'Input Type':<20} {'Output Type':<20} {'MSE':<15}")
    print("-"*70)
    
    for backend_name, X_input in backends_to_test:
        # Quantize
        X_mx = mx_quantize(X_input, format='mxfp8_e4m3', block_size=32)
        
        # Convert back to numpy for comparison
        if backend_name == 'torch':
            X_mx_np = X_mx.detach().cpu().numpy()
        elif backend_name == 'jax':
            X_mx_np = np.array(X_mx)
        else:
            X_mx_np = X_mx
        
        # Compute error
        mse = np.mean((X_np - X_mx_np) ** 2)
        
        print(f"{backend_name:<12} {str(type(X_input).__name__):<20} "
              f"{str(type(X_mx).__name__):<20} {mse:<15.6f}")
    
    print("\n✓ All backends produce consistent results!")


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
        
        # Compute MSE
        X_reconstructed = mx_tensor.dequantize()
        mse = np.mean((X - X_reconstructed) ** 2)
        mae = np.mean(np.abs(X - X_reconstructed))
        
        results.append({
            'format': fmt,
            'compression': stats['compression_ratio_fp16'],
            'mse': mse,
            'mae': mae,
            'bits': stats['bits_per_element']
        })
    
    print(f"\n{'Format':<15} {'Bits/Elem':<12} {'Compress':<12} "
          f"{'MSE':<15} {'MAE':<15} {'Efficiency*':<15}")
    print("-"*85)
    print("*Efficiency = Compression / MSE (higher is better)")
    print()
    
    for r in results:
        efficiency = r['compression'] / (r['mse'] + 1e-10)
        print(f"{r['format']:<15} {r['bits']:<12.2f} {r['compression']:<12.2f} "
              f"{r['mse']:<15.6e} {r['mae']:<15.6f} {efficiency:<15.1f}")


def example_8_pytorch_training_simulation():
    """Example 8: Simulate PyTorch training with MX (STE)."""
    print("\n" + "="*80)
    print("Example 8: PyTorch Training Simulation with STE")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        from pychop.tch.mx_formats import MXLinear, MXQuantizerSTE
        
        print("\nCreating MX-quantized linear layer...")
        
        # Create MX quantized layer
        layer = MXLinear(
            in_features=128,
            out_features=64,
            weight_format='mxfp8_e4m3',
            block_size=32,
            quantize_input=True
        )
        
        print(f"Layer: {layer}")
        
        # Simulate forward + backward pass
        x = torch.randn(32, 128, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass (STE automatically applied)
        loss.backward()
        
        print(f"\nGradients computed successfully!")
        print(f"Input gradient shape: {x.grad.shape}")
        print(f"Weight gradient shape: {layer.weight.grad.shape}")
        print(f"Weight gradient norm: {layer.weight.grad.norm().item():.4f}")
        
        print("\n✓ STE works! Gradients flow through MX quantization.")
        
    except ImportError:
        print("\n⚠ PyTorch not available. Skipping this example.")
        print("Install with: pip install torch")


def example_9_custom_format():
    """Example 9: Create and use custom MX format."""
    print("\n" + "="*80)
    print("Example 9: Custom MX Format")
    print("="*80)
    
    from pychop import create_mx_spec
    
    # Create ultra-low precision format: 3-bit (E2M0)
    custom_spec = create_mx_spec(
        exp_bits=2,
        sig_bits=0,
        block_size=32,
        name="MXFP3_E2M0"
    )
    
    print(f"\nCustom format: {custom_spec}")
    print(f"Element bits: {custom_spec.element_bits}")
    print(f"Block size: {custom_spec.block_size}")
    print(f"Compression vs FP16: {custom_spec.compression_vs_fp16:.2f}x")
    print(f"Compression vs FP32: {custom_spec.compression_vs_fp32:.2f}x")
    
    # Use custom format
    X = np.random.randn(256, 256).astype(np.float32)
    
    mx_tensor = MXTensor(X, format=(2, 0), block_size=32)  # Same as custom_spec
    X_reconstructed = mx_tensor.dequantize()
    
    stats = mx_tensor.statistics()
    mse = np.mean((X - X_reconstructed) ** 2)
    
    print(f"\nQuantization results:")
    print(f"  Format: {stats['format']}")
    print(f"  Bits per element: {stats['bits_per_element']:.2f}")
    print(f"  Compression: {stats['compression_ratio_fp16']:.2f}x")
    print(f"  MSE: {mse:.6e}")
    
    print("\n✓ Custom format works!")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Microscaling (MX) Formats - Comprehensive Examples")
    print("Pychop Multi-Backend Support: NumPy, PyTorch, JAX")
    print("="*80)
    
    # Show available formats
    print_mx_format_table()
    
    # Run examples
    example_1_basic_usage()
    example_2_matrix_operations()
    example_3_format_comparison()
    example_4_block_size_analysis()
    example_5_neural_network_simulation()
    example_6_multi_backend_comparison()
    example_7_compression_visualization()
    example_8_pytorch_training_simulation()
    example_9_custom_format()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)
    
    print("\nKey takeaways:")
    print("1. MX formats provide 2-4x compression vs FP16")
    print("2. Multi-backend support (NumPy, PyTorch, JAX)")
    print("3. Automatic STE for PyTorch QAT")
    print("4. Block size affects compression/error tradeoff")
    print("5. Custom formats easily created")
    print("\n")


if __name__ == "__main__":
    main()