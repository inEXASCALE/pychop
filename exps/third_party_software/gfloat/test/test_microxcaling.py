# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest

import numpy as np
from numpy.typing import NDArray

import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    DTYPE_FP4,
)

from gfloat import (
    BlockFormatInfo,
    FormatInfo,
    RoundMode,
    quantize_block,
    compute_scale_amax,
    encode_block,
)
from gfloat.formats import (
    # format_info_ocp_int8,
    format_info_ocp_e3m2,
    format_info_ocp_e2m1,
    format_info_ocp_e8m0,
)


@pytest.mark.parametrize(
    ("mx_etype,gf_etype"),
    [
        # (ElemFormat.int8, format_info_ocp_int8),
        (DTYPE_FP6_E3M2, format_info_ocp_e3m2),
        (DTYPE_FP4, format_info_ocp_e2m1),
    ],
)
@pytest.mark.parametrize(
    "A",
    [
        np.arange(32) / 2 - 5,
        np.zeros(32),
    ],
    ids=[
        "tennish",
        "zeros",
    ],
)
def test_mx(
    mx_etype: str,
    gf_etype: FormatInfo,
    A: NDArray[np.float64],
) -> None:
    ta = torch.tensor(A, dtype=torch.float32)

    # MX: Quantize
    mx_dq = MXTensor.to_mx(ta, mx_etype).to_dtype(ta.dtype)

    # GFloat: Declare block format
    fi = BlockFormatInfo("test", gf_etype, 32, format_info_ocp_e8m0)

    # GFloat: Quantize
    gf_dq = quantize_block(fi, ta, compute_scale_amax)  # type: ignore [arg-type]

    # Compare
    np.testing.assert_allclose(gf_dq, mx_dq)


def test_mx_exceptions() -> None:
    fi = BlockFormatInfo("test", format_info_ocp_e2m1, 32, format_info_ocp_e8m0)

    A = np.ones(32) * 2.0**-139

    s = compute_scale_amax(fi.etype.emax, A)
    assert s == 2.0**-127

    with pytest.raises(ValueError, match="out of range"):
        list(encode_block(fi, fi.stype.max * 2, A))

    assert not fi.stype.is_signed
    scale = fi.stype.min / 2
    assert scale != 0
    with pytest.raises(ValueError, match="out of range"):
        list(encode_block(fi, scale, A))
