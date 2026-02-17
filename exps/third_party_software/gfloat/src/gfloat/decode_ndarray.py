# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from types import ModuleType
import numpy as np
import numpy.typing as npt
from .types import FormatInfo, Domain


def decode_ndarray(
    fi: FormatInfo, codes: npt.NDArray, np: ModuleType = np
) -> npt.NDArray:
    r"""
    Vectorized version of :meth:`decode_float`

    Args:
      fi (FormatInfo): Floating point format descriptor.
      i (array of int):  Integer code points, in the range :math:`0 \le i < 2^{k}`,
                where :math:`k` = ``fi.k``

    Returns:
      Decoded float values

    Raises:
      ValueError:
        If any :paramref:`i` is outside the range of valid code points in :paramref:`fi`.
    """
    assert np.issubdtype(codes.dtype, np.integer)

    k = fi.k
    p = fi.precision
    t = p - 1  # Trailing significand field width
    num_signbits = 1 if fi.is_signed else 0
    w = k - t - num_signbits  # Exponent field width

    if np.any(codes < 0) or np.any(codes >= 2**k):
        raise ValueError(f"Code point not in range [0, 2**{k})")

    if fi.is_signed:
        signmask = 1 << (k - 1)
        sign = np.where(codes & signmask, -1.0, 1.0)
    else:
        signmask = None
        sign = 1.0

    exp = ((codes >> t) & ((1 << w) - 1)).astype(np.int64)
    significand = codes & ((1 << t) - 1)
    if fi.is_twos_complement:
        significand = np.where(sign < 0, (1 << t) - significand, significand)

    bias = fi.bias

    fval = np.zeros_like(codes, dtype=np.float64)
    isspecial = np.zeros_like(codes, dtype=bool)

    if fi.domain == Domain.Extended:
        fval = np.where(codes == fi.code_of_posinf, np.inf, fval)
        isspecial |= codes == fi.code_of_posinf
        if fi.is_signed:
            fval = np.where(codes == fi.code_of_neginf, -np.inf, fval)
            isspecial |= codes == fi.code_of_neginf

    if fi.num_nans > 0:
        code_is_nan = codes == fi.code_of_nan
        if w > 0:
            # All-bits-special exponent (ABSE)
            abse = exp == 2**w - 1
            min_code_with_nan = 2 ** (p - 1) - fi.num_high_nans
            code_is_nan |= abse & (significand >= min_code_with_nan)

        fval = np.where(code_is_nan, np.nan, fval)
        isspecial |= code_is_nan

    # Zero
    iszero = ~isspecial & (exp == 0) & (significand == 0) & fi.has_zero
    fval = np.where(iszero, 0.0, fval)
    if fi.has_nz:
        fval = np.where(iszero & (sign < 0), -0.0, fval)

    issubnormal = (exp == 0) & (significand != 0) & fi.has_subnormals
    expval = np.where(issubnormal, 1 - bias, exp - bias)
    fsignificand = np.where(issubnormal, 0.0, 1.0) + np.ldexp(significand, -t)

    # Normal/Subnormal/Zero case, other values will be overwritten
    expval_safe = np.where(isspecial | iszero, 0, expval)
    fval_finite_safe = sign * np.ldexp(fsignificand, expval_safe)
    fval = np.where(~(iszero | isspecial), fval_finite_safe, fval)

    return fval
