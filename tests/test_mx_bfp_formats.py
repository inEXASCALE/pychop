import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from pychop.np.bfp_formats import BFPTensor_, bfp_quantize
from pychop.np.mx_formats import MXTensor_, mx_quantize


def test_mxfp4_uses_ocp_scale_for_small_blocks():
    x = np.full(32, 0.25, dtype=np.float32)

    q = mx_quantize(x, format="mxfp4_e2m1")
    mx = MXTensor_(x, format="mxfp4_e2m1")

    np.testing.assert_array_equal(q, x)
    assert mx.blocks[0].shared_scale == pytest.approx(2.0**-4)


def test_mxfp4_rounds_ties_to_even():
    x = np.array([1.25, 1.75, 4.0] + [0.0] * 29, dtype=np.float32)

    q = mx_quantize(x, format="mxfp4_e2m1")

    np.testing.assert_array_equal(q[:3], np.array([1.0, 2.0, 4.0], dtype=np.float32))


def test_mxfp4_supports_ocp_subnormal_elements():
    x = np.array([0.5, 1.0, 6.0] + [0.0] * 29, dtype=np.float32)

    q = mx_quantize(x, format="mxfp4_e2m1")

    np.testing.assert_array_equal(q[:3], x[:3])


def test_mxint8_uses_ocp_twos_complement_scaling_and_ties_to_even():
    x = np.array([1.0, 1.0 / 128.0, 3.0 / 128.0] + [0.0] * 29, dtype=np.float32)

    q = mx_quantize(x, format="mxint8")

    expected = np.array([1.0, 0.0, 1.0 / 32.0], dtype=np.float32)
    np.testing.assert_array_equal(q[:3], expected)


def test_predefined_mx_format_honors_block_size_override():
    mx = MXTensor_(np.ones(33, dtype=np.float32), format="mxfp4_e2m1", block_size=16)

    stats = mx.statistics()
    assert stats["block_size"] == 16
    assert stats["num_blocks"] == 3


def test_bfp_shared_exponent_avoids_saturating_block_maximum():
    x = np.full(32, 1.5, dtype=np.float32)

    q = bfp_quantize(x, format="bfp8")
    bfp = BFPTensor_(x, format="bfp8")

    np.testing.assert_array_equal(q, x)
    assert bfp.blocks[0].shared_exponent == 1


def test_bfp_rounds_ties_to_even():
    x = np.array([1.5, 1.0 / 128.0, 3.0 / 128.0] + [0.0] * 29, dtype=np.float32)

    q = bfp_quantize(x, format="bfp8")

    expected = np.array([1.5, 0.0, 1.0 / 32.0], dtype=np.float32)
    np.testing.assert_array_equal(q[:3], expected)


def test_torch_backend_matches_numpy_for_mx_and_bfp():
    torch = pytest.importorskip("torch")

    x_np = np.linspace(-5.0, 10.5, 32, dtype=np.float32)
    x_torch = torch.tensor(x_np)

    from pychop.tch.bfp_formats import bfp_quantize as torch_bfp_quantize
    from pychop.tch.mx_formats import mx_quantize as torch_mx_quantize

    np.testing.assert_array_equal(
        torch_mx_quantize(x_torch, format="mxfp4_e2m1").detach().cpu().numpy(),
        mx_quantize(x_np, format="mxfp4_e2m1"),
    )
    np.testing.assert_array_equal(
        torch_bfp_quantize(x_torch, format="bfp8").detach().cpu().numpy(),
        bfp_quantize(x_np, format="bfp8"),
    )


def test_jax_backend_matches_numpy_for_mx_and_bfp_without_flax():
    jnp = pytest.importorskip("jax.numpy")

    x_np = np.linspace(-5.0, 10.5, 32, dtype=np.float32)
    x_jax = jnp.asarray(x_np)

    from pychop.jx.bfp_formats import bfp_quantize as jax_bfp_quantize
    from pychop.jx.mx_formats import mx_quantize as jax_mx_quantize

    np.testing.assert_array_equal(
        np.array(jax_mx_quantize(x_jax, format="mxfp4_e2m1")),
        mx_quantize(x_np, format="mxfp4_e2m1"),
    )
    np.testing.assert_array_equal(
        np.array(jax_bfp_quantize(x_jax, format="bfp8")),
        bfp_quantize(x_np, format="bfp8"),
    )
