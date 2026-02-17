# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import array_api_strict as xp
import numpy as np
import pytest

from gfloat import (
    RoundMode,
    FormatInfo,
    decode_float,
    decode_ndarray,
    round_float,
    round_ndarray,
)
from gfloat.formats import *

xp.set_array_api_strict_flags(api_version="2024.12")


@pytest.mark.parametrize("fi", sample_formats)
@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("sat", [True, False])
def test_array_api(fi: FormatInfo, rnd: RoundMode, sat: bool) -> None:
    a0 = np.random.rand(23, 1, 34) - 0.5
    a = xp.asarray(a0)

    srnumbits = 32
    srbits0 = np.random.randint(0, 2**srnumbits, a.shape)
    srbits = xp.asarray(srbits0)

    round_ndarray(fi, a, rnd, sat, srbits=srbits, srnumbits=srnumbits)  # type: ignore
