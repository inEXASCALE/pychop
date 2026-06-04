"""
Test CADNA-style random rounding (rmode=7) across all backends.
"""

import numpy as np
import pytest


def test_cadna_random_generator():
    """Test CADNA random generator produces valid bits."""
    from pychop.cadna_random import CADNARandomGenerator
    
    gen = CADNARandomGenerator(seed=42)
    
    # Generate 10000 bits
    bits = np.array([gen.random_bit() for _ in range(10000)])
    
    # Check properties
    assert bits.min() == 0
    assert bits.max() == 1
    assert 0.45 < bits.mean() < 0.55  # Should be close to 0.5
    print("✓ CADNA random generator test passed")
    

def test_numpy_bit_flip():
    """Test NumPy bit flip implementation."""
    from pychop.cadna_random import numpy_bit_flip, CADNARandomGenerator
    
    gen = CADNARandomGenerator(seed=42)
    
    # Test with known pattern
    x = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float64)
    random_bits = np.array([0, 0, 1, 1], dtype=np.uint8)
    
    x_flipped = numpy_bit_flip(x.copy(), random_bits)
    
    # Expected: [1.5, -2.3, -3.7, 0.5]
    expected = np.array([1.5, -2.3, -3.7, 0.5])
    
    print("Float64 test:")
    print(f"  Original: {x}")
    print(f"  Random bits: {random_bits} (1=flip, 0=keep)")
    print(f"  Flipped: {x_flipped}")
    print(f"  Expected: {expected}")
    print(f"  Match: {np.allclose(x_flipped, expected)}")
    
    np.testing.assert_allclose(x_flipped, expected)
    print("✓ NumPy bit flip test passed")


def test_cadna_rounding_numpy():
    """Test CADNA rounding with NumPy backend - DIRECT."""
    # ============ 修改：直接使用底层类 ============
    from pychop.np.float_point import Chop_
    
    # Create chopper with rmode=7 (directly)
    ch = Chop_(prec='h', rmode=7, random_state=42)
    # ===========================================
    
    # Test data
    x = np.array([1.234, -5.678, 0.999, -0.001])
    
    print(f"Testing with direct Chop_ class")
    print(f"Generator: {ch._cadna_gen}")
    print(f"Initial counter: {ch._cadna_gen._cache_counter}")
    
    # Apply rounding multiple times
    results = []
    for i in range(10):
        result = ch(x.copy())
        results.append(result)
        print(f"Run {i+1}: {result} (counter: {ch._cadna_gen._cache_counter})")
    
    # Check output shape
    assert results[0].shape == x.shape
    
    # Check values are finite
    assert np.all(np.isfinite(results[0]))
    
    # Results should vary (stochastic)
    results_array = np.array(results)
    variances = results_array.var(axis=0)
    
    print(f"\nVariances: {variances}")
    print(f"Has randomness: {np.any(variances > 0)}")
    
    assert np.any(variances > 0), "CADNA rounding should produce varying results"
    print("✓ CADNA rounding (NumPy) test passed")


def test_cadna_rounding_torch():
    """Test CADNA rounding with PyTorch backend - DIRECT."""
    try:
        import torch
        from pychop.tch.float_point import Chop_
    except ImportError:
        pytest.skip("PyTorch not available")
    
    ch = Chop_(prec='h', rmode=7, random_state=42)
    
    x = torch.tensor([1.234, -5.678, 0.999, -0.001])
    
    results = []
    for _ in range(10):
        result = ch(x.clone())
        results.append(result)
    
    results_tensor = torch.stack(results)
    variances = results_tensor.var(dim=0)
    
    assert torch.any(variances > 0), "CADNA should produce varying results"
    print("✓ CADNA rounding (PyTorch) test passed")


def test_cadna_rounding_jax():
    """Test CADNA rounding with JAX backend - DIRECT."""
    try:
        import jax.numpy as jnp
        from pychop.jx.float_point import Chop_
    except ImportError:
        pytest.skip("JAX not available")
    
    ch = Chop_(prec='h', rmode=7, random_state=42)
    
    x = jnp.array([1.234, -5.678, 0.999, -0.001])
    
    results = []
    for _ in range(10):
        result = ch(x)
        results.append(result)
    
    results_array = jnp.stack(results)
    variances = jnp.var(results_array, axis=0)
    
    assert jnp.any(variances > 0), "CADNA should produce varying results"
    print("✓ CADNA rounding (JAX) test passed")


def test_cadna_vs_standard_rounding():
    """Compare CADNA rounding (rmode=7) with standard stochastic (rmode=5)."""
    from pychop.np.float_point import Chop_
    
    ch_cadna = Chop_(prec='s', rmode=7, random_state=42)
    ch_stoc = Chop_(prec='s', rmode=5, random_state=42)
    
    x = np.random.randn(100)
    
    # Run multiple times
    results_cadna = np.array([ch_cadna(x.copy()) for _ in range(50)])
    results_stoc = np.array([ch_stoc(x.copy()) for _ in range(50)])
    
    # Both should have similar mean (around x)
    mean_cadna = results_cadna.mean(axis=0)
    mean_stoc = results_stoc.mean(axis=0)
    
    # Check they're close to original
    np.testing.assert_allclose(mean_cadna, x, rtol=0.1)
    np.testing.assert_allclose(mean_stoc, x, rtol=0.1)
    
    # Check both have variance (stochastic)
    var_cadna = results_cadna.var(axis=0)
    var_stoc = results_stoc.var(axis=0)
    
    assert np.all(var_cadna > 0)
    assert np.all(var_stoc > 0)
    print("✓ CADNA vs standard rounding comparison passed")


# Run tests
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CADNA Rounding Test Suite (Direct Backend Access)")
    print("=" * 60 + "\n")
    
    test_cadna_random_generator()
    print()
    
    test_numpy_bit_flip()
    print()
    
    test_cadna_rounding_numpy()
    print()
    
    try:
        test_cadna_rounding_torch()
        print()
    except:
        print("⊘ PyTorch tests skipped\n")
    
    try:
        test_cadna_rounding_jax()
        print()
    except:
        print("⊘ JAX tests skipped\n")
    
    test_cadna_vs_standard_rounding()
    print()
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)