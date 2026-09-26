"""Shared native tensor algorithm with a small framework operation adapter.

IEEE bit decomposition avoids logarithm errors and handles host subnormal inputs
without depending on a device's floating-point flush-to-zero setting.
"""
import math

from .._validation import bounded_integer
from . import adapter


def _host(ops, dtype, fmt):
    double = dtype == ops.float64
    precision, emin, emax = (53, -1022, 1023) if double else (24, -126, 127)
    if (fmt.precision > precision or 2 - fmt.bias - fmt.precision < emin + 1
            or math.frexp(fmt.max)[1] - 1 > emax - 1):
        raise ValueError("format exceeds this tensor dtype's exact emulation range; use float64 "
                         "(enable jax_enable_x64 for JAX)")
    return (52, 1023, 2047) if double else (23, 127, 255)


def _components(x, ops):
    t, bias = (52, 1023) if x.dtype == ops.float64 else (23, 127)
    raw = ops.bitcast(x)
    magnitude = ops.bit_and(raw, (1 << (t + (11 if t == 52 else 8))) - 1)
    field = ops.right(magnitude, t)
    fraction = ops.bit_and(magnitude, (1 << t) - 1)
    significand = ops.cast(fraction, x.dtype) + ops.cast(ops.where(field > 0, float(1 << t), 0.0), x.dtype)
    exponent = ops.cast(field, ops.int_dtype) - bias
    return raw < 0, magnitude != 0, exponent, significand, t, bias


def quantize(values, fmt, rounding, saturation, srbits, srnumbits, ste, check, name):
    ops = adapter(name)
    x = ops.values(values)
    _host(ops, x.dtype, fmt)
    fast_nearest = name == "torch" and rounding == "nearest_even" and fmt.precision > 1
    if fast_nearest:
        negative, shift, scaled = ops.nearest_components(x, fmt)
    else:
        negative, nonzero, exponent, significand, t, bias = _components(x, ops)
    bits_valid = None
    if rounding.startswith("stochastic_"):
        n = bounded_integer(srnumbits, "srnumbits", 1, 32 if x.dtype == ops.float64 else 23)
        bits = ops.asarray(srbits, x)
        if not ops.integer(bits):
            raise TypeError("srbits must contain integers")
        bits = ops.cast(ops.broadcast(bits, ops.shape(x)), ops.int_dtype)
        bits_valid = (bits >= 0) & (bits <= 2**n - 1)
        if check:
            ops.check(bits_valid, "srbits must lie in [0, 2**srnumbits)")
        bits = ops.cast(bits, x.dtype)
    elif srbits is not None or srnumbits != 0:
        raise ValueError("random bits apply only to stochastic rounding")
    if not fast_nearest:
        shift = ops.maximum(exponent - fmt.precision + 1, 2 - fmt.bias - fmt.precision)
        scale_exp = ops.maximum(exponent, 1 - bias) - t - shift
        scaled = significand * ops.power2(scale_exp, x)
    if rounding == "nearest_even" and fmt.precision > 1:
        rounded = ops.round(scaled)
    else:
        lower = ops.floor(scaled)
        delta = scaled - lower
        inexact = (delta > 0) | ((scaled == 0) & nonzero)
        odd = (lower % 2 != 0) if fmt.precision > 1 else ((lower != 0) & ((shift + fmt.bias) % 2 != 0))
        if rounding == "nearest_even":
            away = (delta > .5) | ((delta == .5) & odd)
        elif rounding == "nearest_away":
            away = delta >= .5
        elif rounding == "toward_zero":
            away = delta < 0  # shape-preserving False
        elif rounding == "toward_positive":
            away = inexact & ~negative
        elif rounding == "toward_negative":
            away = inexact & negative
        elif rounding == "to_odd":
            away = inexact & ~odd
        elif rounding == "stochastic_a":
            away = ops.floor(delta * 2**n) + bits >= 2**n
        elif rounding == "stochastic_b":
            # Compare in half units to avoid a 25-bit sum on float32.
            away = ops.floor(delta * 2**(n + 1)) * .5 + bits >= 2**n - .5
        else:
            away = ops.round(delta * 2**n) + bits >= 2**n
        rounded = lower + ops.cast(away, x.dtype)
    result = ops.rescale(rounded, shift) if fast_nearest else rounded * ops.power2(shift, x)
    # Special values are reconstructed explicitly; bit decomposition never feeds
    # Inf/NaN through integer casts or device-dependent float math.
    result = ops.where(ops.isinf(x), math.inf, result)
    result = ops.where(ops.isnan(x), math.nan, result)
    overflow = result > fmt.max
    if saturation != "none":
        preserve = ops.isinf(x) & (saturation == "propagate") & (fmt.domain == "extended")
        result = ops.where(overflow & ~preserve, fmt.max, result)
    else:
        if rounding in ("toward_zero", "toward_positive", "toward_negative"):
            inward = ops.isfinite(x)
            if rounding == "toward_positive":
                inward = inward & negative
            elif rounding == "toward_negative":
                inward = inward & ~negative
            result = ops.where(overflow & inward, fmt.max, result)
            overflow = overflow & ~inward
        result = ops.where(overflow, math.inf if fmt.domain == "extended" else math.nan, result)
    result = ops.where(negative, -result, result)
    if not fmt.signed:
        invalid_negative = negative & (result != 0)
        result = ops.where(invalid_negative, 0.0 if saturation != "none" else math.nan, result)
        if saturation == "none" and rounding in ("toward_zero", "toward_positive"):
            result = ops.where(invalid_negative & ops.isfinite(x), 0.0, result)
    result = ops.where(result == 0, 0.0, result)
    if bits_valid is not None:
        # Unchecked graph mode still makes invalid elements visible as NaN.
        result = ops.where(bits_valid, result, math.nan)
    result = ops.stop(result)
    return ops.ste(values, result) if ste else result


def encode(values, fmt, rounding, saturation, srbits, srnumbits, check, name):
    ops = adapter(name)
    q = quantize(values, fmt, rounding, saturation, srbits, srnumbits, False, check, name)
    safe = ops.where(ops.isfinite(q), q, 0.0)
    negative, _, exponent, significand, host_t, bias = _components(safe, ops)
    t = fmt.precision - 1
    e = ops.maximum(exponent + fmt.bias, 0)
    scale_exp = ops.maximum(exponent, 1 - bias) - host_t + fmt.bias + t - ops.maximum(e, 1)
    fraction = ops.cast(significand * ops.power2(scale_exp, q), ops.int_dtype)
    code = ops.left(e, t) + fraction - ops.cast(ops.where(e > 0, 1 << t, 0), ops.int_dtype)
    if fmt.domain == "extended":
        code = ops.where(ops.isinf(q), fmt.max_code + 1, code)
    if fmt.signed:
        # sign of q (rather than safe) also distinguishes negative infinity.
        code = code + ops.cast(ops.where(q < 0, 1 << (fmt.k - 1), 0), ops.int_dtype)
    code = ops.where(ops.isnan(q), fmt.nan_code, code)
    return ops.where(q == 0, 0, code)


def decode(codes, fmt, check, name, dtype=None):
    ops = adapter(name)
    if not ops.integer(codes):
        raise TypeError("codes must contain integers")
    if dtype not in (None, "float32", "float64"):
        raise ValueError("dtype must be None, 'float32' or 'float64'")
    if name == "jax" and dtype == "float64" and ops.decode_dtype() == ops.float32:
        raise ValueError("enable jax_enable_x64 before requesting float64 decode")
    dtype = ops.decode_dtype() if dtype is None else getattr(ops, dtype)
    _host(ops, dtype, fmt)
    # Cast first, so an upper bound such as 256 is not converted to uint8 zero.
    code = ops.cast(codes, ops.int_dtype)
    valid = (code >= 0) & (code <= (1 << fmt.k) - 1)
    if check:
        ops.check(valid, "codes must lie in [0, 2**k)")
    # Decode uses float64 where enabled, otherwise the exact supported float32 range.
    magnitude = ops.bit_and(code, (1 << (fmt.k - int(fmt.signed))) - 1)
    t = fmt.precision - 1
    exponent = ops.right(magnitude, t)
    fraction = ops.cast(ops.bit_and(magnitude, (1 << t) - 1), dtype)
    result = (fraction + ops.cast(ops.where(exponent > 0, float(1 << t), 0.0), dtype)) * ops.power2(ops.maximum(exponent, 1) - fmt.bias - t, fraction)
    if fmt.domain == "extended":
        result = ops.where(magnitude == fmt.max_code + 1, math.inf, result)
    if fmt.signed:
        result = ops.where(code >= 1 << (fmt.k - 1), -result, result)
    return ops.where((code == fmt.nan_code) | ~valid, math.nan, result)
