"""Regression tests for MX shared scales at binary exponent boundaries."""

import numpy as np
import pytest

from pychop import mx_quantize
from pychop.np.mx_formats import MXTensor_


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("exponent", (-10, 3, 30))
def test_shared_scale_just_below_and_at_power_of_two(dtype, exponent):
    power = dtype(2.0**exponent)
    below = np.nextafter(power, dtype(0))
    block = np.zeros(32, dtype=dtype)

    block[0] = below
    actual = MXTensor_(block, "mxfp8_e4m3").blocks[0].shared_scale
    expected = 2.0 ** (exponent - 9)
    assert actual == expected, (
        f"MX scale just below 2**{exponent} ({dtype.__name__}): "
        f"expected {expected}, got {actual}"
    )

    block[0] = power
    actual = MXTensor_(block, "mxfp8_e4m3").blocks[0].shared_scale
    expected = 2.0 ** (exponent - 8)
    assert actual == expected, (
        f"MX scale at 2**{exponent} ({dtype.__name__}): "
        f"expected {expected}, got {actual}"
    )


@pytest.mark.parametrize(
    ("format_name", "element_emax"),
    (
        ("mxfp8_e5m2", 15),
        ("mxfp8_e4m3", 8),
        ("mxfp6_e3m2", 4),
        ("mxfp6_e2m3", 2),
        ("mxfp4_e2m1", 2),
        ("mxint8", 0),
    ),
)
def test_all_builtin_formats_use_exact_shared_exponent(format_name, element_emax):
    block = np.zeros(32, dtype=np.float32)
    block[0] = -np.nextafter(np.float32(8), np.float32(0))
    actual = MXTensor_(block, format_name).blocks[0].shared_scale
    expected = 2.0 ** (2 - element_emax)

    assert actual == expected, (
        f"{format_name} scale for a maximum just below 8: "
        f"expected {expected}, got {actual}"
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_mxfp8_e4m3_boundary_quantization(dtype):
    block = np.zeros(32, dtype=dtype)
    block[0] = np.nextafter(dtype(8), dtype(0))
    block[1] = dtype(2.0**-15)

    mx = MXTensor_(block, "mxfp8_e4m3")
    assert mx.blocks[0].shared_scale == 2.0**-6, (
        f"MXFP8 E4M3 boundary scale ({dtype.__name__}) should be 2**-6, "
        f"got {mx.blocks[0].shared_scale}"
    )
    expected = [7.0, 2.0**-15]
    np.testing.assert_array_equal(
        mx.dequantize()[:2], expected,
        err_msg=f"MXTensor_ boundary result differs for {dtype.__name__}",
    )
    np.testing.assert_array_equal(
        mx_quantize(block, "mxfp8_e4m3")[:2], expected,
        err_msg=f"Public mx_quantize boundary result differs for {dtype.__name__}",
    )

    block[0] = dtype(8)
    np.testing.assert_array_equal(
        mx_quantize(block, "mxfp8_e4m3")[:2], [8.0, 0.0],
        err_msg=f"Public mx_quantize result at 8 differs for {dtype.__name__}",
    )

    block[0] = np.nextafter(dtype(8), dtype(np.inf))
    np.testing.assert_array_equal(
        mx_quantize(block, "mxfp8_e4m3")[:2], [8.0, 0.0],
        err_msg=f"Public mx_quantize result just above 8 differs for {dtype.__name__}",
    )


@pytest.mark.parametrize(
    ("maximum", "expected_scale"),
    (
        (0.0, 2.0**-127),
        (np.nextafter(0.0, 1.0), 2.0**-127),
        (np.finfo(np.float64).tiny, 2.0**-127),
        (np.inf, 2.0**127),
        (np.finfo(np.float64).max, 2.0**127),
    ),
)
def test_shared_scale_special_values_and_bounds(maximum, expected_scale):
    block = np.zeros(32, dtype=np.float64)
    block[0] = maximum
    actual = MXTensor_(block, "mxfp8_e4m3").blocks[0].shared_scale
    assert actual == expected_scale, (
        f"MX scale for maximum {maximum}: expected {expected_scale}, got {actual}"
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_all_nan_block_keeps_nan_scale(dtype):
    block = np.full(32, np.nan, dtype=dtype)
    mx = MXTensor_(block, "mxfp8_e4m3")

    assert np.isnan(mx.blocks[0].shared_scale), (
        f"All-NaN MX block ({dtype.__name__}) must keep a NaN shared scale"
    )
    assert np.isnan(mx.dequantize()).all(), (
        f"All-NaN MX block ({dtype.__name__}) must stay NaN after dequantization"
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_nan_does_not_change_finite_block_scale(dtype):
    block = np.zeros(32, dtype=dtype)
    block[0] = np.nan
    block[1] = dtype(8)
    mx = MXTensor_(block, "mxfp8_e4m3")

    assert mx.blocks[0].shared_scale == 2.0**-5, (
        f"Finite maximum must set MX scale even with NaN in {dtype.__name__} block"
    )
    assert np.isnan(mx.dequantize()[0]), "NaN element must remain NaN"
    assert mx.dequantize()[1] == 8.0, "Finite element must remain quantized normally"


@pytest.mark.parametrize("backend", ("torch", "jax", "tensorflow"))
def test_public_mx_quantize_boundary_on_optional_backend(backend):
    block = np.zeros(32, dtype=np.float32)
    block[0] = np.nextafter(np.float32(8), np.float32(0))
    block[1] = 2.0**-15
    expected = np.array([7.0, 2.0**-15], dtype=np.float32)

    if backend == "torch":
        torch = pytest.importorskip("torch")
        result = mx_quantize(torch.from_numpy(block), "mxfp8_e4m3").detach().cpu().numpy()
    elif backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        result = np.asarray(mx_quantize(jnp.asarray(block), "mxfp8_e4m3"))
    else:
        tf = pytest.importorskip("tensorflow")
        result = mx_quantize(tf.convert_to_tensor(block), "mxfp8_e4m3").numpy()

    np.testing.assert_array_equal(
        result[:2], expected,
        err_msg=f"Public MX boundary result differs on the {backend} backend",
    )
