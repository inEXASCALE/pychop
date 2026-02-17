# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import math

import numpy as np

from .types import FormatInfo, Domain


def encode_float(fi: FormatInfo, v: float) -> int:
    """
    Encode input to the given :py:class:`FormatInfo`.

    Will round toward zero if :paramref:`v` is not in the value set.
    Will saturate to `Inf`, `NaN`, `fi.max` in order of precedence.
    Encode -0 to 0 if not `fi.has_nz`

    For other roundings and saturations, call :func:`round_float` first.

    Args:
      fi (FormatInfo): Describes the target format
      v (float): The value to be encoded.

    Returns:
      The integer code point
    """

    # Format Constants
    k = fi.bits
    p = fi.precision
    t = p - 1

    # Encode
    if np.isnan(v):
        return fi.code_of_nan

    # Overflow/underflow
    if v > fi.max:
        if fi.domain == Domain.Extended:
            return fi.code_of_posinf
        if fi.num_nans > 0:
            return fi.code_of_nan
        return fi.code_of_max

    if v < fi.min:
        if fi.domain == Domain.Extended:
            return fi.code_of_neginf
        if fi.num_nans > 0:
            return fi.code_of_nan
        return fi.code_of_min

    # Finite values
    sign = fi.is_signed and np.signbit(v)
    vpos = -v if sign else v

    if fi.has_subnormals and vpos <= fi.smallest_subnormal / 2:
        isig = 0
        biased_exp = 0
    else:
        sig, exp = np.frexp(vpos)
        exp = int(exp)  # All calculations in Python ints

        # sig in range [0.5, 1)
        sig *= 2
        exp -= 1
        # now sig in range [1, 2)

        biased_exp = exp + fi.bias
        if biased_exp < 1 and fi.has_subnormals:
            # subnormal
            sig *= 2.0 ** (biased_exp - 1)
            biased_exp = 0
            assert vpos == sig * 2 ** (1 - fi.bias)
        else:
            if sig > 0:
                sig -= 1.0

        isig = math.floor(sig * 2**t)

    # Zero
    if isig == 0 and biased_exp == 0 and fi.has_zero:
        if sign and fi.has_nz:
            return fi.code_of_negzero
        else:
            return fi.code_of_zero

    # Nonzero
    assert isig < 2**t
    assert biased_exp < 2**fi.expBits or fi.is_twos_complement

    # Handle two's complement encoding
    if fi.is_twos_complement and sign:
        isig = (1 << t) - isig

    # Pack values into a single integer
    code = (int(sign) << (k - 1)) | (biased_exp << t) | (isig << 0)

    return code
