"""Vectorized target-grid quantization for NumPy LightChop."""

import numpy as np


def quantize(x, exp_bits, sig_bits, rmode, subnormal, random_values=None,
             cadna_bits=None):
    """Round a floating array once, then apply target exponent limits."""
    source = np.asarray(x)
    if not np.issubdtype(source.dtype, np.floating):
        source = source.astype(np.float64)
    work_dtype = np.float64 if source.dtype == np.float64 or sig_bits > 23 else np.float32
    work = source.astype(work_dtype, copy=False)
    magnitude = np.abs(work)
    finite = np.isfinite(work)
    safe = np.where(finite, magnitude, 0.0)

    emax = (1 << (exp_bits - 1)) - 1
    emin = 1 - emax
    _, frexp_exp = np.frexp(safe)
    exponent = frexp_exp - 1
    step_exp = np.maximum(exponent, emin) - sig_bits

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = np.ldexp(safe, -step_exp)
    lower = np.floor(scaled)
    fraction = scaled - lower
    upper = lower + 1.0
    inexact = fraction != 0

    if rmode == 1:
        rounded = np.rint(scaled)
    elif rmode == 2:
        rounded = np.where(np.signbit(work), lower, np.where(inexact, upper, lower))
    elif rmode == 3:
        rounded = np.where(np.signbit(work), np.where(inexact, upper, lower), lower)
    elif rmode == 4:
        rounded = lower
    elif rmode == 5:
        rounded = np.where(random_values < fraction, upper, lower)
    elif rmode == 6:
        rounded = np.where(inexact & (random_values >= 0.5), upper, lower)
    elif rmode == 7:
        rounded = np.where(fraction == 0.5, lower, np.rint(scaled))
    elif rmode == 8:
        rounded = np.where(fraction == 0.5, upper, np.rint(scaled))
    elif rmode == 9:
        rounded = np.where(inexact & (np.remainder(lower, 2) == 0), upper, lower)
    elif rmode == 10:
        choose_upper = (cadna_bits == 0) == ~np.signbit(work)
        rounded = np.where(inexact & choose_upper, upper, lower)
    else:
        raise ValueError(f"Unsupported rounding mode: {rmode}")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(rounded, step_exp)
        xmax = np.ldexp(np.float64(2.0 - 2.0 ** -sig_bits), emax)
    if rmode == 2:
        result = np.where((~np.signbit(work)) & (result > xmax), np.inf, result)
        result = np.where(np.signbit(work) & (result > xmax), xmax, result)
    elif rmode == 3:
        result = np.where(np.signbit(work) & (result > xmax), np.inf, result)
        result = np.where((~np.signbit(work)) & (result > xmax), xmax, result)
    elif rmode == 4:
        result = np.minimum(result, xmax)
    else:
        result = np.where(result > xmax, np.inf, result)

    if not subnormal:
        result = np.where(result < np.ldexp(1.0, emin), 0.0, result)

    with np.errstate(invalid="ignore"):
        result = np.copysign(result, work)
        result = np.where(finite, result, work)
    return result.astype(source.dtype, copy=False)
