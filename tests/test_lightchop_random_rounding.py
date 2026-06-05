"""
Test LightChop CADNA-style random rounding rmode=7.

Run:
    pytest tests/test_lightchop_cadna_rmode7.py -s
"""

import importlib
import numpy as np
import pytest


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def available_lightchop_backends():
    backends = ["numpy"]

    if importlib.util.find_spec("torch") is not None:
        backends.append("torch")

    if importlib.util.find_spec("jax") is not None:
        backends.append("jax")

    return backends


def make_fp32_midpoints():
    """
    Construct values exactly halfway between adjacent float32 numbers.

    LightChop with exp_bits=8, sig_bits=23 should simulate fp32.
    """
    base = np.array(
        [
            1.0,
            1.5,
            2.0,
            3.0,
            10.0,
            0.5,
            0.25,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -10.0,
            -0.5,
            -0.25,
        ],
        dtype=np.float32,
    )

    next_up = np.nextafter(base, np.float32(np.inf), dtype=np.float32)

    lower = np.minimum(base, next_up).astype(np.float64)
    upper = np.maximum(base, next_up).astype(np.float64)
    midpoint = (lower + upper) / 2.0

    return midpoint, lower, upper


def make_exact_fp32_values():
    return np.array(
        [
            0.0,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            -2.0,
            10.0,
            -10.0,
            0.25,
            -0.25,
            1024.0,
            -1024.0,
        ],
        dtype=np.float32,
    ).astype(np.float64)


def assert_adjacent_choices(outputs, lower, upper):
    outputs = np.asarray(outputs, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    ulp = np.abs(upper - lower)
    tol = np.maximum(ulp * 0.25, 0.0)

    is_lower = np.abs(outputs - lower) <= tol
    is_upper = np.abs(outputs - upper) <= tol

    bad = ~(is_lower | is_upper)

    if np.any(bad):
        first = tuple(np.argwhere(bad)[0])
        coord = first[-1]
        raise AssertionError(
            "Output is not one of the two adjacent fp32 values.\n"
            f"First bad index: {first}\n"
            f"output={outputs[first]}\n"
            f"lower={lower[coord]}\n"
            f"upper={upper[coord]}\n"
            f"ulp={ulp[coord]}"
        )


def assert_both_directions_seen(outputs, lower, upper):
    outputs = np.asarray(outputs, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    ulp = np.abs(upper - lower)
    tol = np.maximum(ulp * 0.25, 0.0)

    saw_lower = np.any(np.abs(outputs - lower) <= tol, axis=0)
    saw_upper = np.any(np.abs(outputs - upper) <= tol, axis=0)

    assert np.all(saw_lower), f"Never saw lower rounding at coords: {np.where(~saw_lower)[0]}"
    assert np.all(saw_upper), f"Never saw upper rounding at coords: {np.where(~saw_upper)[0]}"


def assert_reasonable_5050_distribution(outputs, lower, upper):
    outputs = np.asarray(outputs, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)

    ulp = np.abs(upper - lower)
    tol = np.maximum(ulp * 0.25, 0.0)

    is_upper = np.abs(outputs - upper) <= tol
    freq_upper = is_upper.mean(axis=0)

    print("  upper frequency:", np.array2string(freq_upper, precision=3))

    # Intentionally loose; catches deterministic or badly biased implementations.
    assert np.all(freq_upper > 0.25)
    assert np.all(freq_upper < 0.75)


# ---------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------

def run_lightchop_backend(backend, x, ntrials=300, seed=42):
    """
    Run LightChop_(exp_bits=8, sig_bits=23, rmode=7) repeatedly.

    exp_bits=8, sig_bits=23 corresponds to float32-style exponent/mantissa
    convention in LightChop.
    """
    x = np.asarray(x, dtype=np.float64)

    if backend == "numpy":
        from pychop.np.lightchop import LightChop_

        ch = LightChop_(
            exp_bits=8,
            sig_bits=23,
            rmode=7,
            subnormal=True,
            random_state=seed,
        )

        outs = []
        for _ in range(ntrials):
            outs.append(np.asarray(ch.quantize(x.copy()), dtype=np.float64))
        return np.stack(outs, axis=0)

    if backend == "torch":
        torch = pytest.importorskip("torch")
        from pychop.tch.lightchop import LightChop_

        ch = LightChop_(
            exp_bits=8,
            sig_bits=23,
            rmode=7,
            subnormal=True,
            random_state=seed,
        )

        x_t = torch.tensor(x, dtype=torch.float64)

        outs = []
        for _ in range(ntrials):
            y = ch.quantize(x_t.clone())
            outs.append(y.detach().cpu().numpy().astype(np.float64))
        return np.stack(outs, axis=0)

    if backend == "jax":
        jax = pytest.importorskip("jax")

        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

        import jax.numpy as jnp
        from pychop.jx.lightchop import LightChop_

        ch = LightChop_(
            exp_bits=8,
            sig_bits=23,
            rmode=7,
            subnormal=True,
            random_state=seed,
        )

        x_j = jnp.asarray(x, dtype=jnp.float64)

        outs = []
        for _ in range(ntrials):
            y = ch(x_j)
            outs.append(np.asarray(y, dtype=np.float64))
        return np.stack(outs, axis=0)

    raise ValueError(f"Unknown backend: {backend}")


def run_lightchop_ste_torch(x, ntrials=300, seed=42, training=True):
    torch = pytest.importorskip("torch")
    from pychop.tch.lightchop import LightChopSTE

    try:
        ch = LightChopSTE(
            exp_bits=8,
            sig_bits=23,
            rmode=7,
            subnormal=True,
            random_state=seed,
        )
    except TypeError:
        pytest.fail(
            "LightChopSTE does not accept random_state. "
            "Update its __init__ signature as suggested."
        )

    if training:
        ch.train()
    else:
        ch.eval()

    x_t = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)

    outs = []
    for _ in range(ntrials):
        y = ch(x_t.clone())
        outs.append(y.detach().cpu().numpy().astype(np.float64))

    return np.stack(outs, axis=0)


# ---------------------------------------------------------------------
# Tests for LightChop_
# ---------------------------------------------------------------------

@pytest.mark.parametrize("backend", available_lightchop_backends())
def test_lightchop_rmode7_midpoints_are_adjacent_choices(backend):
    x_mid, lower, upper = make_fp32_midpoints()

    outputs = run_lightchop_backend(
        backend=backend,
        x=x_mid,
        ntrials=300,
        seed=42,
    )

    print(f"\n[{backend}] LightChop_ adjacent-choice test")
    print("  outputs shape:", outputs.shape)

    assert_adjacent_choices(outputs, lower, upper)


@pytest.mark.parametrize("backend", available_lightchop_backends())
def test_lightchop_rmode7_midpoints_see_both_directions(backend):
    x_mid, lower, upper = make_fp32_midpoints()

    outputs = run_lightchop_backend(
        backend=backend,
        x=x_mid,
        ntrials=300,
        seed=123,
    )

    print(f"\n[{backend}] LightChop_ both-directions test")
    print("  variance:", np.array2string(outputs.var(axis=0), precision=20))

    assert_both_directions_seen(outputs, lower, upper)


@pytest.mark.parametrize("backend", available_lightchop_backends())
def test_lightchop_rmode7_distribution_reasonable(backend):
    x_mid, lower, upper = make_fp32_midpoints()

    outputs = run_lightchop_backend(
        backend=backend,
        x=x_mid,
        ntrials=500,
        seed=777,
    )

    print(f"\n[{backend}] LightChop_ distribution test")
    assert_reasonable_5050_distribution(outputs, lower, upper)


@pytest.mark.parametrize("backend", available_lightchop_backends())
def test_lightchop_rmode7_exact_fp32_values_stable(backend):
    x = make_exact_fp32_values()

    outputs = run_lightchop_backend(
        backend=backend,
        x=x,
        ntrials=100,
        seed=42,
    )

    print(f"\n[{backend}] LightChop_ exact-value stability test")
    print("  max abs error:", np.max(np.abs(outputs - x)))

    np.testing.assert_array_equal(outputs, np.broadcast_to(x, outputs.shape))


@pytest.mark.parametrize("backend", available_lightchop_backends())
def test_lightchop_rmode7_random_state_advances(backend):
    x_mid, _, _ = make_fp32_midpoints()

    outputs = run_lightchop_backend(
        backend=backend,
        x=x_mid,
        ntrials=50,
        seed=999,
    )

    all_identical = np.all(outputs == outputs[0])
    nonzero_var = np.count_nonzero(outputs.var(axis=0) > 0)

    print(f"\n[{backend}] LightChop_ RNG advancement test")
    print("  all outputs identical:", all_identical)
    print("  nonzero variance coords:", nonzero_var, "/", outputs.shape[1])

    assert not all_identical
    assert nonzero_var > 0


# ---------------------------------------------------------------------
# Tests for Torch LightChopSTE
# ---------------------------------------------------------------------

def test_lightchopste_torch_rmode7_midpoints_are_adjacent_choices():
    pytest.importorskip("torch")

    x_mid, lower, upper = make_fp32_midpoints()

    outputs = run_lightchop_ste_torch(
        x=x_mid,
        ntrials=300,
        seed=42,
        training=True,
    )

    print("\n[torch] LightChopSTE adjacent-choice test")
    assert_adjacent_choices(outputs, lower, upper)


def test_lightchopste_torch_rmode7_midpoints_see_both_directions():
    pytest.importorskip("torch")

    x_mid, lower, upper = make_fp32_midpoints()

    outputs = run_lightchop_ste_torch(
        x=x_mid,
        ntrials=300,
        seed=123,
        training=True,
    )

    print("\n[torch] LightChopSTE both-directions test")
    print("  variance:", np.array2string(outputs.var(axis=0), precision=20))

    assert_both_directions_seen(outputs, lower, upper)


def test_lightchopste_torch_rmode7_distribution_reasonable():
    pytest.importorskip("torch")

    x_mid, lower, upper = make_fp32_midpoints()

    outputs = run_lightchop_ste_torch(
        x=x_mid,
        ntrials=500,
        seed=777,
        training=True,
    )

    print("\n[torch] LightChopSTE distribution test")
    assert_reasonable_5050_distribution(outputs, lower, upper)


def test_lightchopste_torch_rmode7_exact_fp32_values_stable():
    pytest.importorskip("torch")

    x = make_exact_fp32_values()

    outputs = run_lightchop_ste_torch(
        x=x,
        ntrials=100,
        seed=42,
        training=True,
    )

    print("\n[torch] LightChopSTE exact-value stability test")
    print("  max abs error:", np.max(np.abs(outputs - x)))

    np.testing.assert_array_equal(outputs, np.broadcast_to(x, outputs.shape))


def test_lightchopste_torch_rmode7_backward_ste_passes_gradient():
    """
    Check STE behavior: forward quantizes, backward passes a gradient.
    This does not assert exact gradient formula, only that backprop works.
    """
    torch = pytest.importorskip("torch")
    from pychop.tch.lightchop import LightChopSTE

    try:
        ch = LightChopSTE(
            exp_bits=8,
            sig_bits=23,
            rmode=7,
            subnormal=True,
            random_state=42,
        )
    except TypeError:
        pytest.fail(
            "LightChopSTE does not accept random_state. "
            "Update its __init__ signature as suggested."
        )

    ch.train()

    x = torch.tensor([1.0, 1.5, -2.0, 3.0], dtype=torch.float64, requires_grad=True)
    y = ch(x)
    loss = y.sum()
    loss.backward()

    print("\n[torch] LightChopSTE backward test")
    print("  y:", y)
    print("  grad:", x.grad)

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-s"])