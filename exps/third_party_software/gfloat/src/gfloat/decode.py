# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np

from .types import FloatClass, FloatValue, FormatInfo, Domain


def decode_float(fi: FormatInfo, i: int) -> FloatValue:
    r"""
    Given :py:class:`FormatInfo` and integer code point, decode to a :py:class:`FloatValue`

    Args:
      fi (FormatInfo): Floating point format descriptor.
      i (int):  Integer code point, in the range :math:`0 \le i < 2^{k}`,
                where :math:`k` = ``fi.k``

    Returns:
      Decoded float value

    Raises:
      ValueError:
        If :paramref:`i` is outside the range of valid code points in :paramref:`fi`.
    """
    assert isinstance(i, int)

    k = fi.k
    p = fi.precision
    t = p - 1  # Trailing significand field width
    num_signbits = 1 if fi.is_signed else 0
    w = k - t - num_signbits  # Exponent field width

    if i < 0 or i >= 2**k:
        raise ValueError(f"Code point {i} not in range [0, 2**{k})")

    if fi.is_signed:
        signmask = 1 << (k - 1)
        signbit = 1 if i & signmask else 0
        sign = -1 if signbit else 1
    else:
        signmask = None
        signbit = 0
        sign = 1

    exp = (i >> t) & ((1 << w) - 1)
    significand = i & ((1 << t) - 1)
    if fi.is_twos_complement and signbit:
        significand = (1 << t) - significand

    bias = fi.bias

    iszero = exp == 0 and significand == 0 and fi.has_zero
    issubnormal = fi.has_subnormals and (exp == 0) and (significand != 0)
    isnormal = not iszero and not issubnormal
    if iszero or issubnormal:
        expval = 1 - bias
        fsignificand = significand * 2**-t
    else:
        expval = exp - bias
        fsignificand = 1.0 + significand * 2**-t

    # Handle specials: Infs, NaN, -0, NaN_0

    # High NaNs
    fval = None
    max_positive_code = (1 << (k - fi.signBits)) - 1
    code_without_sign = i & max_positive_code
    if code_without_sign > max_positive_code - fi.num_high_nans:
        # Return nan, ignore sign
        fval = np.nan

    # Infinities
    if fi.domain == Domain.Extended:
        if code_without_sign == max_positive_code - fi.num_high_nans:
            fval = -np.inf if signbit else np.inf

    # Negative zero or NaN
    if iszero and i == signmask and not fi.is_twos_complement:
        if fi.has_nz:
            fval = -0.0
        else:
            fval = np.nan

    # In range - compute value
    if fval is None:
        fval = sign * fsignificand * 2.0**expval

    # Compute FloatClass
    fclass = None
    if fval == 0:
        fclass = FloatClass.ZERO
    elif np.isnan(fval):
        fclass = FloatClass.NAN
    elif np.isfinite(fval):
        if isnormal:
            fclass = FloatClass.NORMAL
        else:
            fclass = FloatClass.SUBNORMAL
    else:
        fclass = FloatClass.INFINITE

    return FloatValue(i, fval, exp, expval, significand, fsignificand, signbit, fclass)
