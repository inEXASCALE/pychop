"""Regression checks for binary-format conversion in Bitchop."""

import warnings

import numpy as np
import pytest

from pychop import Bitchop


BACKENDS = ("numpy", "torch", "jax", "tensorflow")

ISSUE_15_CASES = (
    pytest.param(0.0, 0x0000, id="positive-zero"),
    pytest.param(-0.0, 0x8000, id="negative-zero"),
    pytest.param(2 - 2**-12, 0x4000, id="significand-carry"),
    pytest.param(1 + 2**-11 + 2**-20, 0x3C01, id="sticky-bit-above-halfway"),
    pytest.param(2**-24, 0x0001, id="smallest-subnormal"),
    pytest.param(3 * 2**-24, 0x0003, id="subnormal-leading-bit"),
    pytest.param(1e-5, 0x00A8, id="subnormal-rounding"),
    pytest.param(70000.0, 0x7C00, id="finite-overflow"),
)


def _result(backend, values, *, dtype=np.float64, format_bits=(5, 10), **kwargs):
    pytest.importorskip({"torch": "torch", "jax": "jax", "tensorflow": "tensorflow"}.get(backend, "numpy"))
    array = np.asarray(values, dtype=dtype)
    result = Bitchop(*format_bits, **kwargs)(array)
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)


# Issue #15: keep each reported failure as a separately named test case.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_value,expected_bits", ISSUE_15_CASES)
def test_issue_15_binary16_regressions(backend, input_value, expected_bits, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    result = _result(backend, [input_value])
    actual_bits = int(result.astype(np.float16).view(np.uint16)[0])
    assert actual_bits == expected_bits, (
        f"{backend} converted {input_value!r} to binary16 0x{actual_bits:04X}; "
        f"expected 0x{expected_bits:04X}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_binary16_nearest_even_matches_numpy(backend, dtype, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    rng = np.random.default_rng(15)
    values = np.concatenate((
        [0.0, -0.0, 2 - 2**-12, 1 + 2**-11 + 2**-20,
         2**-24, 3 * 2**-24, 1e-5, -1e-30, 65504, 65519, 65520,
         -65520, 70000, np.inf, -np.inf, np.nan],
        rng.uniform(-70000, 70000, 64),
        rng.uniform(-1e-4, 1e-4, 64),
    )).astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = values.astype(np.float16).astype(dtype)
    actual = _result(backend, values, dtype=dtype)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_mode_and_unit_roundoff(backend, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    result = _result(backend, [0.0, 1.0, 2 - 2**-12])
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])
    obj = Bitchop(5, 10)
    assert obj.u == 2**-11
    with pytest.raises(ValueError, match="Unknown rounding mode"):
        Bitchop(5, 10, rmode="not_a_mode")


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize("dtype,exp_bits", ((np.float32, 11), (np.float64, 12)))
def test_rejects_target_range_wider_than_result_dtype(backend, dtype, exp_bits, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    pytest.importorskip({"torch": "torch", "jax": "jax", "tensorflow": "tensorflow"}.get(backend, "numpy"))
    value = np.array([np.finfo(dtype).max], dtype=dtype)
    with pytest.raises(ValueError, match="Target exponent width exceeds"):
        Bitchop(exp_bits, 10)(value)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode,expected", (
    ("nearest_even", [1.0, 1 + 2**-10, 1.0, 1 + 2**-9]),
    ("nearest_odd", [1 + 2**-10, 1 + 2**-10, 1.0, 1 + 2**-10]),
    ("plus_infinity", [1 + 2**-10, 1 + 2**-10, 1 + 2**-10, 1 + 2**-9]),
    ("minus_infinity", [1.0, 1.0, 1.0, 1 + 2**-10]),
    ("toward_zero", [1.0, 1.0, 1.0, 1 + 2**-10]),
))
def test_guard_sticky_and_ties(backend, rmode, expected, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    values = [1 + 2**-11, 1 + 2**-11 + 2**-20, 1 + 2**-20, 1 + 3 * 2**-11]
    # The second value lies just above a tie; the third has only a sticky bit.
    np.testing.assert_array_equal(_result(backend, values, rmode=rmode), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode,expected", (
    (1, [np.inf, -np.inf, 65504.0, np.inf]),
    (0, [np.inf, -np.inf, 65504.0, 65504.0]),
    (2, [np.inf, -65504.0, np.inf, np.inf]),
    (3, [65504.0, -np.inf, 65504.0, 65504.0]),
    (4, [65504.0, -65504.0, 65504.0, 65504.0]),
))
def test_overflow_respects_rounding_direction(backend, rmode, expected, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    values = [70000.0, -70000.0, 65519.0, 65520.0]
    np.testing.assert_array_equal(_result(backend, values, rmode=rmode), expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_subnormal_rounding_and_signed_zero(backend, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    values = [-0.0, 0.0, 2**-25, -2**-25, 3 * 2**-25, 2**-24, 3 * 2**-24]
    result = _result(backend, values)
    expected = [-0.0, 0.0, 0.0, -0.0, 2**-23, 2**-24, 3 * 2**-24]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    odd = _result(backend, [2**-25, 3 * 2**-25], rmode="nearest_odd")
    np.testing.assert_array_equal(odd, [2**-24, 2**-24])
    flushed = _result(backend, [-2**-24, 2**-24, 2**-14], subnormal=False)
    np.testing.assert_array_equal(flushed, [0.0, 0.0, 2**-14])
    np.testing.assert_array_equal(np.signbit(flushed), [True, False, False])
    below_min_normal = np.nextafter(2**-14, 0.0)
    np.testing.assert_array_equal(
        _result(backend, [below_min_normal], subnormal=False), [2**-14]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_source_subnormals(backend, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    values = np.array([2**-149, 2**-130, -2**-130, 2**-127], dtype=np.float32)
    exact = _result(backend, values, dtype=np.float32, format_bits=(8, 23))
    np.testing.assert_array_equal(exact.view(np.uint32), values.view(np.uint32))
    bfloat = _result(backend, values, dtype=np.float32, format_bits=(8, 7))
    np.testing.assert_array_equal(bfloat, [0.0, 2**-130, -2**-130, 2**-127])
    directed = _result(backend, [-2**-149], dtype=np.float32,
                       format_bits=(8, 7), rmode="minus_infinity")
    np.testing.assert_array_equal(directed, [-2**-133])


@pytest.mark.parametrize("backend", BACKENDS)
def test_float64_source_subnormals(backend, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    smallest = np.nextafter(0.0, 1.0)
    values = np.array([smallest, -smallest, 2**-1023, 2**-1022])
    result = _result(backend, values, format_bits=(11, 52))
    np.testing.assert_array_equal(result.view(np.uint64), values.view(np.uint64))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_component_metadata_matches_saturation(backend, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    pytest.importorskip({"torch": "torch", "jax": "jax"}.get(backend, "numpy"))
    obj = Bitchop(5, 10, rmode="toward_zero")
    obj(np.array([70000.0, -70000.0, 0.0, 2**-24]))
    def host(array):
        return np.asarray(array.detach().cpu() if hasattr(array, "detach") else array)
    np.testing.assert_array_equal(host(obj.sign), [0, 1, 0, 0])
    np.testing.assert_array_equal(host(obj.exponent), [30, 30, 0, 0])
    np.testing.assert_array_equal(host(obj.significand), [1023, 1023, 0, 1])
    np.testing.assert_array_equal(host(obj.is_denormal), [False, False, False, True])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode", (5, 6))
def test_stochastic_modes_preserve_exact_values_and_seed(backend, rmode, monkeypatch):
    monkeypatch.setenv("chop_backend", backend)
    values = [0.0, -0.0, 1.0, 2**-24, 1 + 2**-11] * 4
    first = _result(backend, values, rmode=rmode, random_state=8)
    second = _result(backend, values, rmode=rmode, random_state=8)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[::5], 0.0)
    np.testing.assert_array_equal(np.signbit(first[1::5]), True)
    np.testing.assert_array_equal(first[2::5], 1.0)
    np.testing.assert_array_equal(first[3::5], 2**-24)
