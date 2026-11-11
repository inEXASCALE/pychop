"""Regression checks for zero signs at floating-point quantization boundaries."""

import numpy as np
import pytest

from pychop import Chop, FaultChop, P3109, P3109Format, mx_quantize


BACKENDS = ("numpy", "torch", "jax", "tensorflow")


def _backend_array(backend, values):
    if backend == "numpy":
        return np.asarray(values, dtype=np.float64)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(values, dtype=torch.float64)
    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return jnp.asarray(values, dtype=jnp.float32)
    tf = pytest.importorskip("tensorflow")
    return tf.constant(values, dtype=tf.float64)


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode", (1, 2, 3, 4))
def test_chop_keeps_negative_zero_and_negative_underflow(backend, rmode, monkeypatch):
    monkeypatch.setenv("chop_backend", "auto")
    x = _backend_array(backend, [-0.0, -1e-30, 0.0])
    result = _numpy(Chop(5, 10, rmode=rmode)(x))
    expected = [-0.0, -2**-24 if rmode == 3 else -0.0, 0.0]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), [True, True, False])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode", (1, 2, 3, 4))
def test_faultchop_keeps_zero_sign_and_finite_results(backend, rmode, monkeypatch):
    monkeypatch.setenv("chop_backend", "auto")
    x = _backend_array(backend, [-0.0, -1e-30, 0.0, 1.0])
    result = _numpy(FaultChop("h", rmode=rmode)(x))
    assert result[0] == 0 and np.signbit(result[0])
    assert result[2] == 0 and not np.signbit(result[2])
    assert result[3] == 1.0
    if rmode == 3:
        assert result[1] < 0
    else:
        assert result[1] == 0 and np.signbit(result[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_disabling_subnormals_keeps_the_sign_of_flushed_values(backend, monkeypatch):
    monkeypatch.setenv("chop_backend", "auto")
    x = _backend_array(backend, [-0.0, -1e-8, 0.0, 1.0])
    for quantizer in (Chop(5, 10, subnormal=False), FaultChop("h", subnormal=False)):
        result = _numpy(quantizer(x))
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(np.signbit(result), [True, True, False, False])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("format_name", ("mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_e2m1"))
def test_mx_keeps_zero_sign_in_mixed_and_zero_blocks(backend, format_name, monkeypatch):
    monkeypatch.setenv("chop_backend", "auto")
    mixed = _backend_array(backend, [-0.0, -1e-30] + [1.0] * 30)
    mixed_result = _numpy(mx_quantize(mixed, format_name))
    np.testing.assert_array_equal(mixed_result[:2], [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(mixed_result[:2]), [True, True])
    np.testing.assert_array_equal(mixed_result[2:], np.ones(30))

    zeros = _backend_array(backend, [-0.0, 0.0] * 16)
    zero_result = _numpy(mx_quantize(zeros, format_name))
    np.testing.assert_array_equal(np.signbit(zero_result), [True, False] * 16)


def test_torch_ste_preserves_zero_sign_and_identity_gradient():
    torch = pytest.importorskip("torch")
    from pychop.tch.lightchop import LightChopSTE

    x = torch.tensor([-0.0, -1e-30, 1.0], dtype=torch.float64, requires_grad=True)
    result = LightChopSTE(5, 10)(x)
    np.testing.assert_array_equal(np.signbit(result.detach().cpu().numpy()), [True, True, False])
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_numpy_internal_rounder_preserves_signed_zero():
    from pychop.np.float_point import _chop_round_to_nearest

    result = _chop_round_to_nearest(np.array([-0.0, -1e-30]), t=11, emax=15)
    np.testing.assert_array_equal(result, [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [True, True])


@pytest.mark.parametrize("rmode", (5, 6, 7, 8, 9, 10))
def test_chop_numpy_other_rounding_modes_keep_exact_zero(rmode):
    result = Chop(5, 10, rmode=rmode)(np.array([-0.0, 0.0]))
    np.testing.assert_array_equal(result, [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [True, False])


@pytest.mark.parametrize("rmode", (5, 6))
def test_faultchop_numpy_stochastic_modes_keep_exact_zero(rmode):
    result = FaultChop("h", rmode=rmode)(np.array([-0.0, 0.0]))
    np.testing.assert_array_equal(result, [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [True, False])


def test_chop_numpy_chunked_path_keeps_negative_zero():
    x = np.ones(32)
    x[:2] = [-0.0, -1e-30]
    result = Chop(5, 10, chunk_size=4)(x)
    np.testing.assert_array_equal(np.signbit(result[:2]), [True, True])
    np.testing.assert_array_equal(result[2:], x[2:])


def test_p3109_canonical_zero_policy_is_unchanged():
    result = P3109(P3109Format())(np.array([-0.0, -1e-30]))
    assert np.all(result == 0)
    assert not np.any(np.signbit(result))
