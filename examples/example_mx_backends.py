"""
MX Formats - Multi-Backend Demo

Demonstrates OCP Microscaling formats across NumPy, JAX, and PyTorch backends.

Usage:
    python examples/example_mx_backends.py --backend auto
    python examples/example_mx_backends.py --backend numpy
    python examples/example_mx_backends.py --backend torch
    python examples/example_mx_backends.py --backend jax
"""

import argparse
import pychop


def test_numpy_backend():
    """Test MX with NumPy backend."""
    print("\n" + "="*80)
    print("Test 1: NumPy Backend")
    print("="*80)
    
    import numpy as np
    from pychop import mx_quantize, MXTensor
    
    pychop.backend('numpy', verbose=True)
    
    X = np.random.randn(1024, 768).astype(np.float32)
    print(f"Input: {X.shape}, dtype={X.dtype}")
    
    # Quantize with MXFP8 E4M3
    X_q = mx_quantize(X, format='mxfp8_e4m3')
    print(f"Output: {X_q.shape}, dtype={X_q.dtype}")
    
    mse = np.mean((X - X_q) ** 2)
    print(f"MSE: {mse:.2e}")
    
    # Statistics
    mx = MXTensor(X, format='mxfp8_e4m3')
    stats = mx.statistics()
    print(f"Compression: {stats['compression_ratio_fp16']:.2f}x vs FP16")
    print(f"✓ NumPy backend works!")


def test_torch_backend():
    """Test MX with PyTorch backend (with STE)."""
    print("\n" + "="*80)
    print("Test 2: PyTorch Backend (with STE)")
    print("="*80)
    
    import torch
    from pychop import mx_quantize
    from pychop.tch.mx_formats import MXQuantizerSTE
    
    pychop.backend('torch', verbose=True)
    
    X = torch.randn(128, 768, requires_grad=True)
    print(f"Input: {X.shape}, requires_grad={X.requires_grad}")
    
    # Quantize (automatic STE!)
    X_q = mx_quantize(X, format='mxfp8_e4m3')
    print(f"Output: {X_q.shape}, requires_grad={X_q.requires_grad}")
    
    # Backward pass
    loss = X_q.sum()
    loss.backward()
    
    print(f"Gradient: {X.grad.shape}")
    print(f"Gradient norm: {X.grad.norm():.2e}")
    print(f"✓ PyTorch backend works with STE!")
    
    # Test quantizer
    print("\nTesting MXQuantizerSTE...")
    quantizer = MXQuantizerSTE(format='mxfp8_e4m3')
    X2 = torch.randn(64, 512, requires_grad=True)
    X2_q = quantizer(X2)
    loss2 = X2_q.sum()
    loss2.backward()
    print(f"✓ MXQuantizerSTE works!")


def test_jax_backend():
    """Test MX with JAX backend."""
    print("\n" + "="*80)
    print("Test 3: JAX Backend (with custom VJP)")
    print("="*80)
    
    try:
        import jax
        import jax.numpy as jnp
        from pychop import mx_quantize
        from pychop.jx.mx_formats import MXQuantizerSTE
        
        pychop.backend('jax', verbose=True)
        
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (256, 512))
        print(f"Input: {X.shape}, dtype={X.dtype}")
        
        X_q = mx_quantize(X, format='mxfp8_e4m3')
        print(f"Output: {X_q.shape}, dtype={X_q.dtype}")
        
        # Test gradient
        print("\nTesting gradient flow...")
        quantizer = MXQuantizerSTE(format='mxfp8_e4m3')
        
        def loss_fn(x):
            x_q = quantizer(x)
            return jnp.sum(x_q ** 2)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(X)
        
        print(f"Gradient: {grads.shape}")
        print(f"Gradient norm: {jnp.linalg.norm(grads):.2e}")
        print(f"✓ JAX backend works with custom VJP!")
        
    except ImportError as e:
        print(f"⚠ JAX not available: {e}")


def test_auto_detection():
    """Test automatic backend detection."""
    print("\n" + "="*80)
    print("Test 4: Automatic Backend Detection")
    print("="*80)
    
    pychop.backend('auto', verbose=True)
    from pychop import mx_quantize
    
    # NumPy
    print("\n1. NumPy input:")
    import numpy as np
    X_np = np.random.randn(100, 100).astype(np.float32)
    X_q_np = mx_quantize(X_np, format='mxfp8_e4m3')
    print(f"Input: {type(X_np)}, Output: {type(X_q_np)}")
    print(f"✓ Auto-detected NumPy backend")
    
    # PyTorch
    print("\n2. PyTorch input:")
    import torch
    X_torch = torch.randn(100, 100)
    X_q_torch = mx_quantize(X_torch, format='mxfp8_e4m3')
    print(f"Input: {type(X_torch)}, Output: {type(X_q_torch)}")
    print(f"✓ Auto-detected PyTorch backend")
    
    # JAX
    try:
        print("\n3. JAX input:")
        import jax.numpy as jnp
        X_jax = jnp.array(np.random.randn(100, 100))
        X_q_jax = mx_quantize(X_jax, format='mxfp8_e4m3')
        print(f"Input: {type(X_jax)}, Output: {type(X_q_jax)}")
        print(f"✓ Auto-detected JAX backend")
    except ImportError:
        print("\n3. JAX not available (skipped)")
    
    print(f"\n✓ Automatic backend detection works!")


def test_format_comparison():
    """Compare different MX formats."""
    print("\n" + "="*80)
    print("Test 5: MX Format Comparison")
    print("="*80)
    
    import numpy as np
    from pychop import compare_mx_formats
    
    pychop.backend('numpy')
    
    X = np.random.randn(512, 512).astype(np.float32)
    compare_mx_formats(X, formats=['mxfp8_e5m2', 'mxfp8_e4m3', 'mxfp6_e3m2', 'mxfp4_e2m1'])


def test_format_table():
    """Print MX format table."""
    print("\n" + "="*80)
    print("Test 6: MX Format Table")
    print("="*80)
    
    from pychop import print_mx_format_table
    print_mx_format_table()


def main():
    parser = argparse.ArgumentParser(description='MX Multi-Backend Demo')
    parser.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['auto', 'numpy', 'torch', 'jax', 'all'],
        help='Backend to test'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MX Formats - Multi-Backend Demo")
    print("="*80)
    
    if args.backend == 'all' or args.backend == 'auto':
        test_auto_detection()
    
    if args.backend == 'all' or args.backend == 'numpy':
        test_numpy_backend()
    
    if args.backend == 'all' or args.backend == 'torch':
        test_torch_backend()
    
    if args.backend == 'all' or args.backend == 'jax':
        test_jax_backend()
    
    test_format_comparison()
    test_format_table()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()