"""
Custom MX Formats Examples

This demonstrates the full flexibility of custom microscaling formats.
"""

import numpy as np
from pychop.mx_formats import MXTensor


def explore_exp_sig_tradeoff():
    """Explore the tradeoff between exponent and significand bits."""
    print("\n" + "="*80)
    print("Exploring Exponent vs Significand Tradeoff")
    print("="*80)
    
    # Generate test data with varying magnitudes
    X_small = np.random.randn(1024) * 0.1  # Small values
    X_large = np.random.randn(1024) * 100  # Large values
    X_mixed = np.concatenate([
        np.random.randn(512) * 0.01,
        np.random.randn(512) * 10
    ])  # Mixed magnitudes
    
    # Test different E/M combinations with same total bits (8-bit elements)
    formats = [
        (5, 2),  # More exponent (better for large dynamic range)
        (4, 3),  # Balanced
        (3, 4),  # More significand (better for precision)
        (2, 5),  # Maximum significand
    ]
    
    datasets = {
        'Small values': X_small,
        'Large values': X_large,
        'Mixed range': X_mixed
    }
    
    for data_name, data in datasets.items():
        print(f"\n{data_name}:")
        print(f"{'Format':<10} {'Exp':<6} {'Sig':<6} {'Mean Error':<15} {'Max Error':<15} {'Rel Error':<15}")
        print("-"*75)
        
        for exp, sig in formats:
            mx = MXTensor(data, format=(exp, sig), block_size=32)
            stats = mx.statistics()
            
            print(f"E{exp}M{sig}{'':<6} {exp:<6} {sig:<6} "
                  f"{stats['mean_abs_error']:<15.6f} "
                  f"{stats['max_abs_error']:<15.6f} "
                  f"{stats['mean_rel_error']:<15.4%}")


def extreme_low_precision():
    """Test extreme low-precision formats (2-3 bit elements)."""
    print("\n" + "="*80)
    print("Extreme Low-Precision Formats")
    print("="*80)
    
    X = np.random.randn(1024).astype(np.float32)
    
    extreme_formats = [
        (1, 2),  # 4-bit: E1M2
        (2, 1),  # 4-bit: E2M1
        (1, 1),  # 3-bit: E1M1 (!!!)
        (1, 0),  # 2-bit: E1M0 (only sign + exponent!)
    ]
    
    print(f"\n{'Format':<10} {'Bits':<8} {'Block Size':<12} {'Compression':<15} "
          f"{'Mean Error':<15} {'Usable?':<10}")
    print("-"*75)
    
    for exp, sig in extreme_formats:
        element_bits = 1 + exp + sig
        
        # Try different block sizes
        best_result = None
        for block_size in [16, 32, 64]:
            try:
                mx = MXTensor(X, format=(exp, sig), block_size=block_size)
                stats = mx.statistics()
                
                if best_result is None or stats['mean_abs_error'] < best_result['error']:
                    best_result = {
                        'block_size': block_size,
                        'compression': stats['compression_ratio_fp16'],
                        'error': stats['mean_abs_error'],
                        'usable': stats['mean_rel_error'] < 0.5  # Less than 50% error
                    }
            except:
                pass
        
        if best_result:
            usable = "✓ Yes" if best_result['usable'] else "✗ No"
            print(f"E{exp}M{sig}{'':<6} {element_bits:<8} {best_result['block_size']:<12} "
                  f"{best_result['compression']:<15.2f} "
                  f"{best_result['error']:<15.6f} {usable:<10}")


def custom_scale_formats():
    """Explore custom scale factor formats."""
    print("\n" + "="*80)
    print("Custom Scale Factor Formats")
    print("="*80)
    
    X = np.random.randn(2048) * np.random.choice([0.01, 1, 100], size=2048)
    
    print("\nEffect of scale factor precision on extreme dynamic range data:")
    print(f"Data range: [{X.min():.2e}, {X.max():.2e}]")
    
    # Fixed element format (E4M3), varying scale
    scale_configs = [
        (6, 0),   # E6M0 scale (small dynamic range)
        (8, 0),   # E8M0 scale (standard)
        (10, 0),  # E10M0 scale (large dynamic range)
        (8, 1),   # E8M1 scale (with mantissa)
        (8, 2),   # E8M2 scale (more mantissa)
    ]
    
    print(f"\n{'Scale Format':<15} {'Scale Bits':<12} {'Total Overhead*':<15} "
          f"{'Mean Error':<15} {'Max Error':<15}")
    print("-"*75)
    print("*Overhead = (scale_bits / block_size) per element")
    print()
    
    for scale_exp, scale_sig in scale_configs:
        mx = MXTensor(
            X, 
            format=(4, 3),  # E4M3 elements
            block_size=32,
            scale_exp_bits=scale_exp,
            scale_sig_bits=scale_sig
        )
        stats = mx.statistics()
        
        scale_bits = 1 + scale_exp + scale_sig
        overhead = scale_bits / 32  # per element
        
        print(f"E{scale_exp}M{scale_sig}{'':<11} {scale_bits:<12} {overhead:<15.3f} "
              f"{stats['mean_abs_error']:<15.6f} "
              f"{stats['max_abs_error']:<15.6f}")


def block_size_optimization():
    """Find optimal block size for different formats."""
    print("\n" + "="*80)
    print("Block Size Optimization")
    print("="*80)
    
    X = np.random.randn(4096).astype(np.float32)
    
    formats_to_test = [
        (4, 3, "E4M3 (8-bit)"),
        (5, 4, "E5M4 (10-bit)"),
        (2, 1, "E2M1 (4-bit)"),
    ]
    
    block_sizes = [8, 16, 32, 64, 128, 256]
    
    for exp, sig, name in formats_to_test:
        print(f"\n{name}:")
        print(f"{'Block Size':<12} {'Compression':<15} {'Mean Error':<15} "
              f"{'Storage (bytes)':<18} {'Efficiency**':<15}")
        print("-"*80)
        print("**Efficiency = Compression / (Error * Storage_KB)")
        print()
        
        for bs in block_sizes:
            mx = MXTensor(X, format=(exp, sig), block_size=bs)
            stats = mx.statistics()
            
            storage_kb = stats['storage_bytes'] / 1024
            efficiency = stats['compression_ratio_fp16'] / (stats['mean_abs_error'] * storage_kb + 1e-10)
            
            print(f"{bs:<12} {stats['compression_ratio_fp16']:<15.2f} "
                  f"{stats['mean_abs_error']:<15.6f} "
                  f"{stats['storage_bytes']:<18.1f} {efficiency:<15.1f}")


def application_specific_formats():
    """Design formats for specific applications."""
    print("\n" + "="*80)
    print("Application-Specific Format Design")
    print("="*80)
    
    # Scenario 1: Weight matrices (static, need precision)
    print("\n1. Neural Network Weights (static, need precision):")
    W = np.random.randn(512, 512).astype(np.float32) * 0.1
    
    weight_formats = [
        ((5, 5), "E5M5 - High precision"),
        ((4, 4), "E4M4 - Balanced"),
        ((3, 5), "E3M5 - Max significand"),
    ]
    
    print(f"{'Format':<25} {'Storage (KB)':<15} {'Compression':<15} {'Mean Error':<15}")
    print("-"*70)
    
    for fmt, desc in weight_formats:
        mx = MXTensor(W, format=fmt, block_size=64)
        stats = mx.statistics()
        print(f"{desc:<25} {stats['storage_bytes']/1024:<15.2f} "
              f"{stats['compression_ratio_fp32']:<15.2f} "
              f"{stats['mean_abs_error']:<15.6f}")
    
    # Scenario 2: Activations (dynamic, need range)
    print("\n2. Neural Network Activations (dynamic range):")
    A = np.maximum(0, np.random.randn(1024, 512) * 10)  # ReLU output
    
    activation_formats = [
        ((5, 3), "E5M3 - Max exponent"),
        ((4, 4), "E4M4 - Balanced"),
        ((6, 2), "E6M2 - Even more range"),
    ]
    
    print(f"{'Format':<25} {'Storage (KB)':<15} {'Compression':<15} {'Max Error':<15}")
    print("-"*70)
    
    for fmt, desc in activation_formats:
        mx = MXTensor(A, format=fmt, block_size=32)
        stats = mx.statistics()
        print(f"{desc:<25} {stats['storage_bytes']/1024:<15.2f} "
              f"{stats['compression_ratio_fp16']:<15.2f} "
              f"{stats['max_abs_error']:<15.6f}")
    
    # Scenario 3: Gradients (sparse, need dynamic range)
    print("\n3. Gradients (sparse, high dynamic range):")
    G = np.random.randn(1024, 512) * 0.001
    G[np.random.rand(1024, 512) > 0.9] *= 1000  # Some large gradients
    
    gradient_formats = [
        ((6, 2), "E6M2 - Huge range"),
        ((5, 3), "E5M3 - Large range"),
        ((4, 4), "E4M4 - Balanced"),
    ]
    
    print(f"{'Format':<25} {'Storage (KB)':<15} {'Compression':<15} {'Rel Error':<15}")
    print("-"*70)
    
    for fmt, desc in gradient_formats:
        mx = MXTensor(G, format=fmt, block_size=32)
        stats = mx.statistics()
        print(f"{desc:<25} {stats['storage_bytes']/1024:<15.2f} "
              f"{stats['compression_ratio_fp16']:<15.2f} "
              f"{stats['mean_rel_error']:<15.4%}")


def main():
    """Run all custom format examples."""
    print("\n" + "="*80)
    print("Custom MX Formats - Advanced Examples")
    print("="*80)
    print("\nThese examples show how to design custom MX formats for specific needs.")
    print("You have full control over:")
    print("  - Element format (exp_bits, sig_bits)")
    print("  - Scale format (scale_exp_bits, scale_sig_bits)")
    print("  - Block size")
    print("="*80)
    
    explore_exp_sig_tradeoff()
    extreme_low_precision()
    custom_scale_formats()
    block_size_optimization()
    application_specific_formats()
    
    print("\n" + "="*80)
    print("Key Takeaways:")
    print("="*80)
    print("1. More exponent bits → better for large dynamic range")
    print("2. More significand bits → better for precision")
    print("3. Larger blocks → better compression (but may reduce precision)")
    print("4. Custom scale formats can help with extreme ranges")
    print("5. Different applications need different tradeoffs!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
