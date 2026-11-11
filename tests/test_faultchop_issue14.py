"""Issue #14: target-grid rounding, format parameters, and OCP E4M3."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pychop import Customs, FaultChop


BACKENDS = ("numpy", "torch", "jax", "tensorflow")


def _host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _convert(backend, values, monkeypatch, *, prec="h", dtype=np.float64, **kwargs):
    pytest.importorskip({"torch": "torch", "jax": "jax", "tensorflow": "tensorflow"}.get(backend, "numpy"))
    monkeypatch.setenv("chop_backend", backend)
    array = np.asarray(values, dtype=dtype)
    if backend == "torch":
        import torch
        array = torch.as_tensor(array.copy())
    elif backend == "jax":
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        array = jnp.asarray(array)
    elif backend == "tensorflow":
        import tensorflow as tf
        array = tf.convert_to_tensor(array)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = FaultChop(prec, **kwargs)(array)
    return np.asarray(_host(result))


ISSUE_14_CASES = (
    pytest.param(5e-324, 0.0, {}, "h", id="smallest-float64-to-half-zero"),
    pytest.param(1e-310, 0.0, {}, "h", id="tiny-float64-to-half-zero"),
    pytest.param(np.nextafter(2**-25, np.inf), 2**-24, {}, "h", id="above-half-smallest-subnormal"),
    pytest.param(2**-25, 0.0, {}, "h", id="exact-half-smallest-subnormal-tie-even"),
    pytest.param(2**-24, 2**-24, {}, "h", id="smallest-subnormal-exact"),
    pytest.param(1e-5, 1.0013580322265625e-5, {}, "h", id="subnormal-leading-bit"),
    pytest.param(70000.0, np.inf, {"customs": Customs(exp_bits=5, sig_bits=10)}, "h", id="custom-exponent-overflow"),
    pytest.param(448.0, 448.0, {}, "fp8-e4m3", id="ocp-largest-finite"),
    pytest.param(480.0, 448.0, {}, "fp8-e4m3", id="ocp-reserved-nan-code-saturates"),
    pytest.param(448.0, np.inf, {}, "q43", id="legacy-q43-overflows"),
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value,expected,options,prec", ISSUE_14_CASES)
def test_issue_14_reported_conversions(backend, value, expected, options, prec, monkeypatch):
    result = _convert(backend, [value], monkeypatch, prec=prec, **options)
    np.testing.assert_array_equal(result, [expected], err_msg=f"{backend}/{prec}: input={value!r}")


def test_jax_faultchop_accepts_legacy_array_device_api(monkeypatch):
    jax = pytest.importorskip("jax")
    from pychop._jax_compat import enable_x64
    import jax.numpy as jnp
    from pychop.jx.float_point import _single_array_device

    device = jax.devices()[0]

    class LegacyArray:
        def devices(self):
            return {device}

    assert _single_array_device(LegacyArray()) == device
    monkeypatch.setenv("chop_backend", "jax")
    with enable_x64(True):
        source = jnp.asarray([0.007617498318158107, 8.209882269614045e-6])
        result = FaultChop("q43", subnormal=False)(source)
        np.testing.assert_array_equal(np.asarray(result), [0.0, 0.0])
        result = FaultChop("q52", subnormal=True)(source)
        np.testing.assert_array_equal(np.asarray(result)[1], 2**-16)
        converter = FaultChop("q43", subnormal=False)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(converter)(source)),
            np.asarray(converter(source)),
        )


@pytest.mark.parametrize("fixture,index,smallest,expected", (
    ("q43/q43_rmode_1_subnormal_0.mat", 1774, 2**-6, 0.0),
    ("q43/q43_rmode_1_subnormal_1.mat", 2, 2**-9, 2**-9),
    ("q52/q52_rmode_1_subnormal_1.mat", 1387, 2**-16, -2**-16),
))
def test_legacy_mat_rounding_boundary_reference(fixture, index, smallest, expected):
    root = Path(__file__).resolve().parents[1] / "tests"
    recorded = loadmat(root / fixture)["emu_vals"][index, 0]
    input_value = loadmat(root / "verified_data.mat")["array"][0, index]
    if "subnormal_1" in fixture:
        input_value /= 1000
    assert abs(input_value) != smallest / 2
    nearest = np.copysign(
        smallest if abs(input_value) > smallest / 2 else 0.0, input_value
    )
    assert recorded == expected == nearest


@pytest.mark.parametrize("backend", BACKENDS)
def test_issue_14_custom_wide_exponent_subnormal_and_zero(backend, monkeypatch):
    smallest = 2**-1052
    values = [5e-324, 1.036131e-317, smallest, -smallest]
    result = _convert(backend, values, monkeypatch, customs=Customs(t=31, emax=1023))
    np.testing.assert_array_equal(result, [0.0, 0.0, smallest, -smallest])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode,expected", (
    pytest.param(1, [0.0, -0.0], id="nearest-even"),
    pytest.param(2, [2**-24, -0.0], id="toward-plus-infinity"),
    pytest.param(3, [0.0, -2**-24], id="toward-minus-infinity"),
    pytest.param(4, [0.0, -0.0], id="toward-zero"),
))
def test_issue_14_float64_subnormals_in_all_deterministic_modes(
    backend, rmode, expected, monkeypatch
):
    result = _convert(backend, [5e-324, -5e-324], monkeypatch, rmode=rmode)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode", (1, 2, 3, 4))
def test_explim_false_preserves_source_subnormals_and_rounds_at_input_exponent(
    backend, rmode, monkeypatch
):
    values = [5e-324, -5e-324, 1e-310, -1e-310, 1.23456789]
    expected = _convert("numpy", values, monkeypatch, rmode=rmode, explim=False)
    actual = _convert(backend, values, monkeypatch, rmode=rmode, explim=False)
    np.testing.assert_array_equal(actual[:2], values[:2])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_fault_probability_does_not_change_underflow(backend, monkeypatch):
    result = _convert(backend, [5e-324, -5e-324], monkeypatch, flip=True, p=0)
    np.testing.assert_array_equal(result, [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [False, True])


@pytest.mark.parametrize("rmode", (1, 2, 3, 4, 5, 6, 10))
def test_torch_fault_injection_keeps_tiny_values_finite(rmode, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("chop_backend", "torch")
    source = torch.tensor([5e-324, -5e-324] * 64, dtype=torch.float64)
    result = FaultChop("h", rmode=rmode, flip=True, p=0.5, random_state=0)(source)
    assert result.device == source.device
    assert result.dtype == source.dtype
    assert bool(torch.isfinite(result).all()), f"rmode={rmode} produced a nonfinite fault result"
    np.testing.assert_array_equal(np.signbit(result.numpy()), np.signbit(source.numpy()))


def test_torch_fault_probability_and_seed_at_underflow(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("chop_backend", "torch")
    source = torch.full((4096,), -5e-324, dtype=torch.float64)
    first = FaultChop("h", flip=True, p=0.2, random_state=14)(source)
    second = FaultChop("h", flip=True, p=0.2, random_state=14)(source)
    torch.testing.assert_close(first, second)
    rate = float((first != 0).to(torch.float32).mean())
    assert 0.17 < rate < 0.23, f"fault rate {rate:.3f} differs from p=0.2"
    assert bool(torch.isfinite(first).all())
    assert bool(torch.signbit(first).all())


def test_torch_fault_injection_preserves_special_inputs(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("chop_backend", "torch")
    source = torch.tensor([-0.0, 0.0, np.inf, -np.inf, np.nan], dtype=torch.float64)
    result = FaultChop("h", flip=True, p=1, random_state=14)(source)
    np.testing.assert_array_equal(result[:4].numpy(), source[:4].numpy())
    np.testing.assert_array_equal(np.signbit(result[:4].numpy()), [True, False, False, True])
    assert bool(torch.isnan(result[4]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_issue_14_bfloat16_without_subnormals(backend, monkeypatch):
    values = [2**-127 - 2**-140, 2**-127, 2**-126, -2**-127 + 2**-140]
    result = _convert(backend, values, monkeypatch, prec="b")
    np.testing.assert_array_equal(result, [0.0, 2**-126, 2**-126, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [False, False, False, True])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode,expected", (
    pytest.param(1, [0.0, 2**-24, -0.0], id="nearest-even"),
    pytest.param(2, [2**-24, 2**-24, -0.0], id="toward-plus-infinity"),
    pytest.param(3, [0.0, 0.0, -2**-24], id="toward-minus-infinity"),
    pytest.param(4, [0.0, 0.0, -0.0], id="toward-zero"),
))
def test_issue_14_rounding_modes_at_underflow(backend, rmode, expected, monkeypatch):
    values = [2**-25, np.nextafter(2**-25, np.inf), -2**-25]
    result = _convert(backend, values, monkeypatch, rmode=rmode)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("overflow,expected", (
    pytest.param("saturate", [448.0, 448.0, -448.0, 448.0], id="saturate"),
    pytest.param("nan", [448.0, np.nan, np.nan, np.nan], id="nan"),
))
def test_ocp_e4m3_overflow_modes(backend, overflow, expected, monkeypatch):
    result = _convert(backend, [448.0, 465.0, -480.0, np.inf], monkeypatch,
                      prec="fp8-e4m3", overflow=overflow)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tf32_and_bf16_names(backend, monkeypatch):
    for alias, canonical in (("t", "tf32"), ("bf16", "bfloat16")):
        values = [0.0, 1 + 2**-11 + 2**-20, 2**-127 - 2**-140]
        np.testing.assert_array_equal(
            _convert(backend, values, monkeypatch, prec=alias),
            _convert(backend, values, monkeypatch, prec=canonical),
        )


def test_custom_fields_do_not_mutate_caller_and_reject_conflicts(monkeypatch):
    monkeypatch.setenv("chop_backend", "numpy")
    original = Customs(exp_bits=5, sig_bits=10)
    converter = FaultChop(customs=original)
    assert (converter.emax, converter.t) == (15, 11)
    assert (original.emax, original.t) == (None, None)
    with pytest.raises(ValueError, match="emax conflicts with exp_bits"):
        FaultChop(customs=Customs(emax=31, exp_bits=5, t=11))
    with pytest.raises(ValueError, match="t conflicts with sig_bits"):
        FaultChop(customs=Customs(emax=15, t=12, sig_bits=10))


def test_ocp_e4m3_rejects_unsupported_modes_and_legacy_remains_available(monkeypatch):
    monkeypatch.setenv("chop_backend", "numpy")
    with pytest.raises(ValueError, match="use q43"):
        FaultChop("fp8-e4m3", rmode="toward_zero")
    with pytest.raises(ValueError, match="use q43"):
        FaultChop("fp8-e4m3", flip=True)
    with pytest.raises(ValueError, match="available only for fp8-e4m3"):
        FaultChop("q43", overflow="nan")
    assert FaultChop("q43", rmode="toward_zero")(np.array([240.0]))[0] == 240.0


def test_auto_backend_passes_ocp_overflow_policy(monkeypatch):
    monkeypatch.delenv("chop_backend", raising=False)
    converter = FaultChop("fp8-e4m3", overflow="nan")
    result = converter(np.array([448.0, 480.0]))
    assert result[0] == 448.0
    assert np.isnan(result[1])


def test_jax_fast_path_keeps_input_device(monkeypatch):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    monkeypatch.setenv("chop_backend", "jax")
    source = jax.device_put(jnp.asarray([1.0, 2**-24]), jax.devices()[0])
    result = FaultChop("h")(source)
    assert result.devices() == source.devices()
    np.testing.assert_array_equal(np.asarray(result), [1.0, 2**-24])


def test_numpy_private_rounder_does_not_make_nan_on_tiny_values():
    from pychop.np.float_point import _chop_round_to_nearest

    result = _chop_round_to_nearest(np.array([5e-324, -5e-324]), t=11, emax=15)
    np.testing.assert_array_equal(result, [0.0, 0.0])
    np.testing.assert_array_equal(np.signbit(result), [False, True])
