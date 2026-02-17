# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Test that finfo methods on FloatFormat agree with numpy/ml_dtypes

import ml_dtypes
import numpy as np
import pytest

from gfloat import decode_float, round_float
from gfloat.formats import *


@pytest.mark.parametrize(
    "fmt,npfmt",
    [
        (format_info_ocp_e5m2, ml_dtypes.float8_e5m2),
        (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn),
        (format_info_binary16, np.float16),
        (format_info_bfloat16, ml_dtypes.bfloat16),
    ],
)
def test_finfo(fmt: FormatInfo, npfmt: np.dtype) -> None:
    assert fmt.eps == ml_dtypes.finfo(npfmt).eps
    assert fmt.epsneg == ml_dtypes.finfo(npfmt).epsneg
    assert fmt.max == ml_dtypes.finfo(npfmt).max
    assert fmt.maxexp == ml_dtypes.finfo(npfmt).maxexp


def test_constants() -> None:
    assert format_info_p3109(8, 1).smallest_subnormal == 2.0**-63
    assert format_info_p3109(8, 4).smallest_subnormal == 2.0**-10
    assert format_info_p3109(8, 7).smallest_subnormal == 2.0**-6
