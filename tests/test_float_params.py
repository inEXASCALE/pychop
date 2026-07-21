import numpy as np

from pychop import BFP_FORMATS, MX_FORMATS, float_params


def test_float_params_keeps_legacy_tuple_for_binary_formats():
    u, xmins, xmin, xmax, p, emins, emin, emax = float_params("fp16")

    assert u == 2 ** -11
    assert xmins == 2 ** -24
    assert xmin == 2 ** -14
    assert xmax == 65504.0
    assert p == 11
    assert emins == -24
    assert emin == -14
    assert emax == 15


def test_float_params_accepts_supported_binary_aliases():
    np.testing.assert_allclose(float_params("bf16"), float_params("bfloat16"))
    np.testing.assert_allclose(float_params("t"), float_params("tf32"))


def test_float_params_table_includes_registered_bfp_and_mx_formats():
    table = float_params()
    names = set(table[""])

    assert set(BFP_FORMATS).issubset(names)
    assert set(MX_FORMATS).issubset(names)
    assert "flexpoint16" in names
    assert "mxint8" in names


def test_float_params_returns_named_dicts_for_block_formats():
    bfp = float_params("bfp8")
    mx = float_params("mxfp8_e4m3")

    assert bfp["family"] == "bfp"
    assert bfp["block_size"] == BFP_FORMATS["bfp8"].block_size
    assert bfp["scale_bits"] == BFP_FORMATS["bfp8"].exponent_bits

    assert mx["family"] == "mx"
    assert mx["block_size"] == MX_FORMATS["mxfp8_e4m3"].block_size
    assert mx["scale_bits"] == (
        MX_FORMATS["mxfp8_e4m3"].scale_exp_bits
        + MX_FORMATS["mxfp8_e4m3"].scale_sig_bits
    )
