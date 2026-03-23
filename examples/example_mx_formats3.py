"""
Block Floating Point (BFP) Formats - Comprehensive Usage Examples

This script demonstrates BFP quantization with:
1. Basic usage across backends
2. Neural network simulation
3. Compression analysis
4. Comparison with MX formats

"""

import numpy as np
import pychop

from pychop import (
    bfp_quantize,
    BFPTensor,
    BFP_FORMATS,
    print_bfp_format_table
)


def example_1_basic_usage():
    """Example 1: Basic BFP quantization."""
    print("\n" + "="*80)
    print("Example 1: Basic BFP Quantization")
    print("="*80)
    
    pychop.backend('auto')
    
    X = np.random.randn(1024).astype(np.float32)
    
    # Quantize to BFP8
    X_bfp = bfp_quantize(X, format='bfp8')
    
    error = np.abs(X - X_bfp)
    
    print(f"Original min/max:      [{X.min():.4f}, {X.max():.4f}]")
    print(f"Quantized min/max:     [{X_bfp.min():.4f}, {X_bfp.max():.4f}]")
    print(f"Mean absolute error:   {error.mean():.6f}")
    print(f"Max absolute error:    {error.max():.6f}")
    print(f"Mean relative error:   {(error / (np.abs(X) + 1e-10)).mean():.4%}")


def example_2_compression_comparison():
    """Example 2: Compare BFP formats."""
    print("\n" + "="*80)
    print("Example 2: BFP Format Comparison")
    print("="*80)
    
    X = np.random.randn(512, 256).astype(np.float32)
    
    formats = ['bfp16', 'bfp8', 'bfp6', 'bfp4']
    
    print(f"\n{'Format':<15} {'Mantissa':<12} {'Block':<8} {'Compress':<12} "
          f"{'MSE':<15} {'MAE':<15}")
    print("-"*85)
    
    for fmt in formats:
        bfp_tensor = BFPTensor(X, format=fmt)
        X_reconstructed = bfp_tensor.dequantize()
        
        stats = bfp_tensor.statistics()
        mse = np.mean((X - X_reconstructed) ** 2)
        mae = np.mean(np.abs(X - X_reconstructed))
        
        print(f"{stats['format']:<15} {stats['mantissa_bits']:<12} "
              f"{stats['block_size']:<8} {stats['compression_ratio_fp16']:<12.2f} "
              f"{mse:<15.6e} {mae:<15.6f}")


def example_3_neural_network():
    """Example 3: Neural network with BFP."""
    print("\n" + "="*80)
    print("Example 3: Neural Network with BFP")
    print("="*80)
    
    # MLP parameters
    W1 = np.random.randn(256, 128).astype(np.float32) * 0.1
    W2 = np.random.randn(128, 64).astype(np.float32) * 0.1
    X = np.random.randn(32, 256).astype(np.float32)
    
    # FP32 forward
    H1_fp32 = np.maximum(0, X @ W1)
    Y_fp32 = H1_fp32 @ W2
    
    # BFP forward
    W1_bfp = bfp_quantize(W1, format='bfp8')
    W2_bfp = bfp_quantize(W2, format='bfp8')
    X_bfp = bfp_quantize(X, format='bfp8')
    
    H1_bfp = np.maximum(0, X_bfp @ W1_bfp)
    Y_bfp = H1_bfp @ W2_bfp
    
    # Stats
    w1_tensor = BFPTensor(W1, format='bfp8')
    w1_stats = w1_tensor.statistics()
    
    error = np.abs(Y_fp32 - Y_bfp)
    
    print(f"\nNetwork: 256 -> 128 -> 64")
    print(f"Batch size: 32")
    print(f"\nWeight compression: {w1_stats['compression_ratio_fp16']:.2f}x vs FP16")
    print(f"\nOutput error:")
    print(f"  MSE: {np.mean(error**2):.6e}")
    print(f"  MAE: {error.mean():.6f}")
    print(f"  Max: {error.max():.6f}")


def example_4_pytorch_training():
    """Example 4: PyTorch training with BFP."""
    print("\n" + "="*80)
    print("Example 4: PyTorch Training with BFP (STE)")
    print("="*80)
    
    try:
        import torch
        from pychop.tch.bfp_formats import BFPLinear, BFPQuantizerSTE
        
        # Create BFP layer
        layer = BFPLinear(
            in_features=128,
            out_features=64,
            weight_format='bfp8',
            quantize_input=True
        )
        
        print(f"Layer: {layer}")
        
        # Forward + backward
        x = torch.randn(32, 128, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        print(f"\nInput: {x.shape}")
        print(f"Output: {y.shape}")
        print(f"Weight gradient norm: {layer.weight.grad.norm().item():.4f}")
        print("\n✓ STE works with BFP!")
        
    except ImportError:
        print("\n⚠ PyTorch not available")


def example_5_bfp_vs_mx():
    """Example 5: Compare BFP vs MX formats."""
    print("\n" + "="*80)
    print("Example 5: BFP vs MX Comparison")
    print("="*80)
    
    from pychop import mx_quantize, MXTensor
    
    X = np.random.randn(512, 512).astype(np.float32)
    
    # BFP8
    bfp_tensor = BFPTensor(X, format='bfp8')
    X_bfp = bfp_tensor.dequantize()
    bfp_stats = bfp_tensor.statistics()
    bfp_mse = np.mean((X - X_bfp) ** 2)
    
    # MXFP8
    mx_tensor = MXTensor(X, format='mxfp8_e4m3', block_size=32)
    X_mx = mx_tensor.dequantize()
    mx_stats = mx_tensor.statistics()
    mx_mse = np.mean((X - X_mx) ** 2)
    
    print(f"\n{'Format':<20} {'Compression':<15} {'MSE':<15} {'Bits/Elem':<15}")
    print("-"*70)
    print(f"{'BFP8':<20} {bfp_stats['compression_ratio_fp16']:<15.2f} "
          f"{bfp_mse:<15.6e} {bfp_stats['bits_per_element']:<15.2f}")
    print(f"{'MXFP8_E4M3':<20} {mx_stats['compression_ratio_fp16']:<15.2f} "
          f"{mx_mse:<15.6e} {mx_stats['bits_per_element']:<15.2f}")
    
    print("\nKey differences:")
    print("- BFP: One shared exponent per block (simpler hardware)")
    print("- MX:  Each element has exponent + shared scale (better precision)")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Block Floating Point (BFP) - Comprehensive Examples")
    print("="*80)
    
    print_bfp_format_table()
    
    example_1_basic_usage()
    example_2_compression_comparison()
    example_3_neural_network()
    example_4_pytorch_training()
    example_5_bfp_vs_mx()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()