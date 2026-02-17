# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import fractions

import numpy as np


def float_pow2str(v: float, min_exponent: float = -np.inf) -> str:
    """
    Render floating point values as exact fractions times a power of two.

    Example: float_pow2str(127.0) is "127/64*2^6",

    That is (a significand between 1 and 2) times (a power of two).

    If `min_exponent` is supplied, then values with exponent below `min_exponent`,
    are printed as fractions less than 1, with exponent set to `min_exponent`.
    This is typically used to represent subnormal values.

    """
    if not np.isfinite(v):
        return str(v)

    signstr = "-" if np.signbit(v) else ""

    x = np.abs(v)
    e = int(np.floor(np.log2(x)))
    sig = np.ldexp(x, -e)
    if e < min_exponent:
        sig = np.ldexp(sig, e - min_exponent)
        e = int(min_exponent)

    pow2str = f"2^{e:d}"

    significand = fractions.Fraction(sig)
    if significand == 1:
        return signstr + pow2str
    else:
        return signstr + f"{significand}*{pow2str}"


def float_tilde_unless_roundtrip_str(v: float, width: int = 14, d: int = 8) -> str:
    """
    Return a string representation of :paramref:`v`, in base 10,
    with maximum width :paramref:`width` and decimal digits :paramref:`d`


    """
    # valstr: string representation of value in base 10
    # If the representation does not roundtrip to the value,
    # it is preceded by a "~" to indicate "approximately equal to"
    s = f"{v}"
    if len(s) > width:
        if abs(v) < 1 and "e" not in s:
            s = f"{v:.{d}f}"
        else:
            s = f"{v:.{d}}"
    if np.isfinite(v) and float(s) != v:
        s = "~" + s

    return s
