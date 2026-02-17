# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest

from gfloat import (
    decode_float,
    decode_block,
    quantize_block,
    encode_block,
    compute_scale_amax,
)
from gfloat.formats import *


@pytest.mark.parametrize("fi", all_block_formats)
def test_blocks(fi: BlockFormatInfo) -> None:

    vals = np.linspace(-37.0, 42.0, 32)

    scale = compute_scale_amax(fi.etype.emax, vals)
    block = list(encode_block(fi, scale, vals / scale))
    decoded_vals = list(decode_block(fi, block))

    etype_next_under_max = decode_float(fi.etype, fi.etype.code_of_max - 1).fval
    atol = (fi.etype.max - etype_next_under_max) * scale / 2
    np.testing.assert_allclose(decoded_vals, vals, atol=atol)

    via_qb = quantize_block(fi, vals, compute_scale_amax)
    np.testing.assert_allclose(via_qb, decoded_vals, atol=0.0)
