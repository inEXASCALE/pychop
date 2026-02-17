# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np

from gfloat import float_pow2str, float_tilde_unless_roundtrip_str


def test_pow2str() -> None:
    assert float_pow2str(127) == "127/64*2^6"
    assert float_pow2str(1.0625 * 2.0**-12) == "17/16*2^-12"
    assert float_pow2str(3.0 * 2.0**-12) == "3/2*2^-11"
    assert float_pow2str(3.0 / 16 * 2.0**-8) == "3/2*2^-11"
    assert float_pow2str(3.0 / 16 * 2.0**-8, min_exponent=-8) == "3/16*2^-8"


def test_tilde_unless_roundtrip() -> None:
    assert float_tilde_unless_roundtrip_str(1.52587892525e-05) == "~1.5258789e-05"
    assert float_tilde_unless_roundtrip_str(28672.0) == "28672.0"
    assert float_tilde_unless_roundtrip_str(0.0009765625) == "0.0009765625"
    assert float_tilde_unless_roundtrip_str(120.0) == "120.0"
    assert float_tilde_unless_roundtrip_str(0.0010001, width=7, d=4) == "~0.0010"
    assert float_tilde_unless_roundtrip_str(np.inf, width=7, d=4) == "inf"
    assert float_tilde_unless_roundtrip_str(np.nan, width=7, d=4) == "nan"
