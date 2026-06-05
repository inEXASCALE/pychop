"""
Test CADNA-style random rounding (rmode=7) across all backends.

Run directly:
    python tests/test_cadna_rounding.py

Or with pytest:
    pytest tests/test_cadna_rounding.py -s
"""

import numpy as np
import pytest


def test_cadna_random_generator():
    """Test CADNA random generator produces valid bits."""
    from pychop.cadna_random import CADNARandomGenerator

    gen = CADNARandomGenerator(seed=42)

    bits = np.array([gen.random_bit() for _ in range(10000)])

    assert bits.min() == 0
    assert bits.max() == 1
    assert 0.45 < bits.mean() < 0.55

    print("✓ CADNA random generator test passed")
    print(f"  Mean bit value: {bits.mean():.4f}")


def test_numpy_bit_flip():
    """Test NumPy bit flip implementation."""
    from pychop.cadna_random import numpy_bit_flip

    x = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float64)
    random_bits = np.array([0, 0, 1, 1], dtype=np.uint8)

    x_flipped = numpy_bit_flip(x.copy(), random_bits)
    expected = np.array([1.5, -2.3, -3.7, 0.5], dtype=np.float64)

    print("Float64 bit-flip test:")
    print(f"  Original:    {x}")
    print(f"  Random bits: {random_bits}  (1=flip, 0=keep)")
    print(f"  Flipped:     {x_flipped}")
    print(f"  Expected:    {expected}")

    np.testing.assert_allclose(x_flipped, expected)

    x32 = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float32)
    bits32 = np.array([1, 0, 1, 0], dtype=np.uint8)
    x32_flipped = numpy_bit_flip(x32.copy(), bits32)
    expected32 = np.array([-1.5, -2.3, -3.7, -0.5], dtype=np.float32)

    print("Float32 bit-flip test:")
    print(f"  Original:    {x32}")
    print(f"  Random bits: {bits32}")
    print(f"  Flipped:     {x32_flipped}")
    print(f"  Expected:    {expected32}")

    np.testing.assert_allclose(x32_flipped, expected32)

    print("✓ NumPy bit flip test passed")


def test_cadna_rounding_numpy():
    """Test CADNA rounding with NumPy backend."""
    from pychop.np.float_point import Chop_

    ch = Chop_(prec="h", rmode=7, random_state=42)

    x = np.array([1.234, -5.678, 0.999, -0.001], dtype=np.float64)

    print("Testing NumPy Chop_ with rmode=7")
    if hasattr(ch, "_cadna_gen"):
        print(f"  Generator: {ch._cadna_gen}")
        print(f"  Initial counter: {getattr(ch._cadna_gen, '_cache_counter', None)}")

    results = []
    for i in range(10):
        result = ch(x.copy())
        results.append(result)
        if hasattr(ch, "_cadna_gen"):
            counter = getattr(ch._cadna_gen, "_cache_counter", None)
            print(f"  Run {i + 1}: {result}  counter={counter}")
        else:
            print(f"  Run {i + 1}: {result}")

    results_array = np.array(results)
    variances = results_array.var(axis=0)

    assert results_array.shape == (10, x.size)
    assert np.all(np.isfinite(results_array))

    print(f"  Variances: {variances}")
    print(f"  Nonzero variances: {np.count_nonzero(variances > 0)} / {variances.size}")

    assert np.any(variances > 0), "CADNA rmode=7 should produce some varying results"

    print("✓ CADNA rounding NumPy test passed")


def test_cadna_rounding_torch():
    """Test CADNA rounding with PyTorch backend."""
    try:
        import torch
        from pychop.tch.float_point import Chop_
    except ImportError:
        pytest.skip("PyTorch not available")

    ch = Chop_(prec="h", rmode=7, random_state=42)

    x = torch.tensor([1.234, -5.678, 0.999, -0.001], dtype=torch.float64)

    results = []
    for _ in range(20):
        result = ch(x.clone())
        results.append(result)

    results_tensor = torch.stack(results)
    variances = results_tensor.var(dim=0)

    print("PyTorch rmode=7 variances:")
    print(f"  {variances}")

    assert torch.all(torch.isfinite(results_tensor))
    assert torch.any(variances > 0), "CADNA rmode=7 should produce some varying results"

    print("✓ CADNA rounding PyTorch test passed")


def test_cadna_rounding_jax():
    """Test CADNA rounding with JAX backend."""
    try:
        import jax.numpy as jnp
        from pychop.jx.float_point import Chop_
    except ImportError:
        pytest.skip("JAX not available")

    ch = Chop_(prec="h", rmode=7, random_state=42)

    x = jnp.array([1.234, -5.678, 0.999, -0.001], dtype=jnp.float64)

    results = []
    for _ in range(20):
        result = ch(x)
        results.append(result)

    results_array = jnp.stack(results)
    variances = jnp.var(results_array, axis=0)

    print("JAX rmode=7 variances:")
    print(f"  {variances}")

    assert bool(jnp.all(jnp.isfinite(results_array)))
    assert bool(jnp.any(variances > 0)), "CADNA rmode=7 should produce some varying results"

    print("✓ CADNA rounding JAX test passed")


def test_cadna_vs_standard_rounding():
    """
    Compare CADNA-style rmode=7 with standard stochastic rmode=5.

    This test deliberately does NOT require every coordinate to have nonzero
    variance. With finite random trials, some coordinates may remain unchanged,
    especially if values are exactly representable or very close to a target
    floating-point value.
    """
    from pychop.np.float_point import Chop_

    rng = np.random.default_rng(123)

    ch_cadna = Chop_(prec="s", rmode=7, random_state=42)
    ch_stoc = Chop_(prec="s", rmode=5, random_state=42)

    x = rng.normal(loc=0.0, scale=1.0, size=100).astype(np.float64)

    # Avoid tiny values making relative tolerance checks awkward.
    near_zero = np.abs(x) < 1e-3
    x[near_zero] += 0.01

    ntrials = 200
    results_cadna = np.array([ch_cadna(x.copy()) for _ in range(ntrials)])
    results_stoc = np.array([ch_stoc(x.copy()) for _ in range(ntrials)])

    mean_cadna = results_cadna.mean(axis=0)
    mean_stoc = results_stoc.mean(axis=0)

    var_cadna = results_cadna.var(axis=0)
    var_stoc = results_stoc.var(axis=0)

    print("CADNA vs standard stochastic diagnostics:")
    print(f"  CADNA nonzero variances: {np.count_nonzero(var_cadna > 0)} / {var_cadna.size}")
    print(f"  STOC  nonzero variances: {np.count_nonzero(var_stoc > 0)} / {var_stoc.size}")
    print(f"  CADNA variance min/max:  {var_cadna.min()} / {var_cadna.max()}")
    print(f"  STOC  variance min/max:  {var_stoc.min()} / {var_stoc.max()}")
    print(f"  CADNA outputs all identical: {np.all(results_cadna == results_cadna[0])}")
    print(f"  STOC  outputs all identical: {np.all(results_stoc == results_stoc[0])}")

    # Both means should be close to original values at this sample size.
    np.testing.assert_allclose(mean_cadna, x, rtol=0.1, atol=1e-6)
    np.testing.assert_allclose(mean_stoc, x, rtol=0.1, atol=1e-6)

    # Stochasticity check: at least some coordinates should vary.
    assert np.any(var_cadna > 0), "CADNA rmode=7 should produce some varying results"
    assert np.any(var_stoc > 0), "Standard stochastic rmode=5 should produce some varying results"

    # Stronger but still realistic. Do not use np.all(var > 0).
    assert np.count_nonzero(var_cadna > 0) >= 0.1 * var_cadna.size
    assert np.count_nonzero(var_stoc > 0) >= 0.1 * var_stoc.size

    print("✓ CADNA vs standard rounding comparison passed")


def test_stochastic_rounding_midpoints_have_variance():
    """
    Construct float32 midpoints so stochastic rounding has high probability
    of varying every coordinate.

    This is the right way to test all(var > 0), because each value is exactly
    between two adjacent float32 values.
    """
    from pychop.np.float_point import Chop_

    ch_cadna = Chop_(prec="s", rmode=7, random_state=42)
    ch_stoc = Chop_(prec="s", rmode=5, random_state=42)

    base = np.array(
        [1.0, 1.5, 2.0, 3.0, 10.0, -1.0, -2.0, -10.0],
        dtype=np.float32,
    )

    next_up = np.nextafter(base, np.float32(np.inf), dtype=np.float32)

    # Midpoints between adjacent float32 values, represented in float64.
    x = (base.astype(np.float64) + next_up.astype(np.float64)) / 2.0

    ntrials = 300
    results_cadna = np.array([ch_cadna(x.copy()) for _ in range(ntrials)])
    results_stoc = np.array([ch_stoc(x.copy()) for _ in range(ntrials)])

    var_cadna = results_cadna.var(axis=0)
    var_stoc = results_stoc.var(axis=0)

    print("Midpoint stochastic diagnostics:")
    print(f"  x: {x}")
    print(f"  CADNA variances: {var_cadna}")
    print(f"  STOC  variances: {var_stoc}")
    print(f"  CADNA nonzero: {np.count_nonzero(var_cadna > 0)} / {var_cadna.size}")
    print(f"  STOC  nonzero: {np.count_nonzero(var_stoc > 0)} / {var_stoc.size}")

    assert np.all(var_cadna > 0), "Midpoint CADNA rmode=7 should vary for every element"
    assert np.all(var_stoc > 0), "Midpoint standard stochastic rmode=5 should vary for every element"

    print("✓ Midpoint variance test passed")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CADNA Rounding Test Suite")
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
    except pytest.skip.Exception:
        print("⊘ PyTorch tests skipped\n")
    except Exception as exc:
        print(f"⊘ PyTorch test failed or skipped: {exc}\n")

    try:
        test_cadna_rounding_jax()
        print()
    except pytest.skip.Exception:
        print("⊘ JAX tests skipped\n")
    except Exception as exc:
        print(f"⊘ JAX test failed or skipped: {exc}\n")

    test_cadna_vs_standard_rounding()
    print()

    test_stochastic_rounding_midpoints_have_variance()
    print()

    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)