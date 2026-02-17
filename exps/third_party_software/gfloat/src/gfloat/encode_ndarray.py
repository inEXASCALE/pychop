# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from .types import FormatInfo, Domain
import numpy as np
import numpy.typing as npt


def encode_ndarray(fi: FormatInfo, v: npt.NDArray) -> npt.NDArray:
    """
    Vectorized version of :meth:`encode_float`.

    Encode inputs to the given :py:class:`FormatInfo`.

    Will round toward zero if :paramref:`v` is not in the value set.
    Will saturate to `Inf`, `NaN`, `fi.max` in order of precedence.
    Encode -0 to 0 if not `fi.has_nz`

    For other roundings and saturations, call :func:`round_ndarray` first.

    Args:
      fi (FormatInfo): Describes the target format
      v (float array): The value to be encoded.

    Returns:
      The integer code point
    """
    k = fi.bits
    p = fi.precision
    t = p - 1

    sign = np.signbit(v) & fi.is_signed
    vpos = np.where(sign, -v, v)

    nan_mask = np.isnan(v)

    code = np.zeros_like(v, dtype=np.uint64)

    if fi.num_nans > 0:
        code[nan_mask] = fi.code_of_nan
    else:
        assert not np.any(nan_mask)

    if fi.domain == Domain.Extended:
        code[v > fi.max] = fi.code_of_posinf
        if fi.is_signed:
            code[v < fi.min] = fi.code_of_neginf
    else:
        code[v > fi.max] = fi.code_of_nan if fi.num_nans > 0 else fi.code_of_max
        if fi.is_signed:
            code[v < fi.min] = fi.code_of_nan if fi.num_nans > 0 else fi.code_of_min

    if fi.has_zero:
        if fi.has_nz:
            code[v == 0] = np.where(sign[v == 0], fi.code_of_negzero, fi.code_of_zero)
        else:
            code[v == 0] = fi.code_of_zero

    finite_mask = (code == 0) & (v != 0)
    assert not np.any(np.isnan(vpos[finite_mask]))
    if np.any(finite_mask):
        finite_vpos = vpos[finite_mask]
        finite_sign = sign[finite_mask]

        sig, exp = np.frexp(finite_vpos)

        biased_exp = exp.astype(np.int64) + (fi.bias - 1)
        subnormal_mask = (biased_exp < 1) & fi.has_subnormals

        biased_exp_safe = np.where(subnormal_mask, biased_exp, 0)
        tsig = np.where(subnormal_mask, np.ldexp(sig, biased_exp_safe), sig * 2 - 1.0)
        biased_exp[subnormal_mask] = 0

        isig = np.floor(np.ldexp(tsig, t)).astype(np.int64)

        zero_mask = fi.has_zero & (isig == 0) & (biased_exp == 0)
        if not fi.has_nz:
            finite_sign[zero_mask] = False

        # Handle two's complement encoding
        if fi.is_twos_complement:
            isig[finite_sign] = (1 << t) - isig[finite_sign]

        code[finite_mask] = (
            (finite_sign.astype(int) << (k - 1)) | (biased_exp << t) | (isig << 0)
        )

    return code
