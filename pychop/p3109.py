"""Multi-backend emulation of the P3109 public draft scalar formats.

This is a draft-format emulator, not an IEEE conformance claim. Arithmetic
between calls uses the host dtype; call ``quantize`` at every desired rounding
boundary. See docs/source/p3109.rst for the pinned reference and limitations.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from ._validation import bounded_integer as _integer

__all__ = ["P3109Format", "P3109", "p3109_quantize", "p3109_encode", "p3109_decode"]

_MODES = ("nearest_even", "nearest_away", "toward_zero", "toward_positive", "toward_negative", "to_odd",
          "stochastic_a", "stochastic_b", "stochastic_c")


@dataclass(frozen=True)
class P3109Format:
    """Draft format: total width ``k``, precision including the implicit bit.

    Supports 3..32 bits, 1..k precision bits (p < k if signed), signed/unsigned and finite/extended
    domains. At most 10 exponent bits are supported so every finite value and
    midpoint is exactly representable in the float64 computation dtype.
    """

    k: int = 8
    precision: int = 4
    signed: bool = True
    domain: str = "extended"

    def __post_init__(self):
        object.__setattr__(self, "k", _integer(self.k, "k", 3, 32))
        object.__setattr__(self, "precision", _integer(self.precision, "precision", 1, self.k))
        if type(self.signed) is not bool:
            raise TypeError("signed must be a bool")
        if self.signed and self.precision == self.k:
            raise ValueError("signed formats require precision < k")
        if self.domain not in ("finite", "extended"):
            raise ValueError("domain must be 'finite' or 'extended'")
        if self.exponent_bits > 10:
            raise ValueError("at most 10 exponent bits are supported by the float64 host")

    @property
    def name(self):
        return f"p3109_k{self.k}p{self.precision}{'s' if self.signed else 'u'}{self.domain[0]}"

    @property
    def exponent_bits(self):
        return self.k - self.precision + 1 - int(self.signed)

    @property
    def bias(self):
        return (1 << self.exponent_bits) // 2

    @property
    def nan_code(self):
        return 1 << (self.k - 1) if self.signed else (1 << self.k) - 1

    @property
    def max_code(self):
        return (1 << (self.k - int(self.signed))) - 1 - int(not self.signed) - int(self.domain == "extended")

    @property
    def max(self):
        return _decode_magnitude(self.max_code, self)

    @property
    def min(self):
        return -self.max if self.signed else 0.0

    @property
    def smallest_subnormal(self):
        """Smallest positive value (normal when precision is one)."""
        return math.ldexp(1.0, 2 - self.bias - self.precision)

    def to_dict(self):
        """Return JSON-compatible constructor parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, parameters):
        """Restore parameters; unknown fields and invalid values are rejected."""
        return cls(**parameters)


def _decode_magnitude(code, fmt):
    fraction_bits = fmt.precision - 1
    exponent = code >> fraction_bits
    fraction = code & ((1 << fraction_bits) - 1)
    return math.ldexp(float(fraction + ((1 << fraction_bits) if exponent else 0)),
                      max(exponent, 1) - fmt.bias - fraction_bits)


def _format(fmt):
    if not isinstance(fmt, P3109Format):
        raise TypeError("format must be a P3109Format")
    return fmt


def _values(values):
    a = np.asarray(values)
    if a.dtype.kind not in "fiu" or (a.dtype.kind == "f" and a.dtype.itemsize > 8):
        raise TypeError("values must be real float16/32/64 or integer array-like data")
    return a.astype(np.float64, copy=False)


def _saturation(value):
    if type(value) is bool:
        return "finite" if value else "none"
    if isinstance(value, str) and value in ("none", "finite", "propagate"):
        return value
    raise ValueError("saturate must be bool or 'none', 'finite', 'propagate'")


def p3109_quantize(values, fmt=P3109Format(), *, rounding="nearest_even", saturate=False,
                   srbits=None, srnumbits=0, ste=False, check=True):
    """Round data on its native NumPy/Torch/TensorFlow/JAX backend.

    NumPy returns float64; tensors preserve float32/64 (other real dtypes promote
    to float32). Set ste=True for an identity surrogate tensor gradient. Set
    check=False only for prevalidated compiled stochastic inputs; see the guide.

    Six deterministic and three explicit-random-bit stochastic modes are supported. NaNs propagate; zero is always +0.
    Saturation clips infinities and overflow to finite extrema. Otherwise nearest
    overflow produces infinity in extended formats and NaN in finite formats;
    directed rounding toward the finite range clamps finite overflow. Negative
    rounded inputs to unsigned formats become NaN, except inward directed rounding
    or saturation returns zero. The "propagate" policy preserves representable
    infinities while clipping finite overflow.
    """
    fmt = _format(fmt)
    if rounding not in _MODES:
        raise ValueError(f"rounding must be one of {_MODES}")
    saturation = _saturation(saturate)
    if type(ste) is not bool or type(check) is not bool:
        raise TypeError("ste and check must be bools")
    from ._p3109 import backend_of
    tensor_backend = backend_of(values)
    if tensor_backend is not None:
        from ._p3109.kernel import quantize
        return quantize(values, fmt, rounding, saturation, srbits, srnumbits, ste, check, tensor_backend)
    if ste:
        raise ValueError("STE requires a Torch, TensorFlow or JAX floating-point tensor")
    x = _values(values)
    if rounding.startswith("stochastic_"):
        srnumbits = _integer(srnumbits, "srnumbits", 1, 32)
        bits = np.asarray(srbits)
        if bits.dtype.kind not in "iu":
            raise TypeError("srbits must contain unsigned random integers")
        if np.any(bits < 0) or np.any(bits >= 1 << srnumbits):
            raise ValueError("srbits must be in [0, 2**srnumbits)")
        bits = np.broadcast_to(bits, x.shape)
    elif srbits is not None or srnumbits != 0:
        raise ValueError("random bits apply only to stochastic rounding")
    # frexp is exact, including at powers of two; log2 is not safe here.
    magnitude = np.abs(x)
    exponent = np.frexp(magnitude)[1]
    shift = np.maximum(exponent - fmt.precision, 2 - fmt.bias - fmt.precision)
    with np.errstate(invalid="ignore", over="ignore"):
        scaled = np.ldexp(magnitude, -shift)
        if rounding == "nearest_even":
            rounded = np.rint(scaled)
            if fmt.precision == 1:
                # With no trailing significand, tie parity belongs to the exponent code.
                lower = np.floor(scaled)
                odd = (lower != 0) & (((shift + fmt.bias) & 1) != 0)
                rounded = lower + ((scaled - lower > .5) | ((scaled - lower == .5) & odd))
        elif rounding == "nearest_away":
            lower = np.floor(scaled)
            rounded = lower + (scaled - lower >= .5)
        elif rounding == "toward_zero":
            rounded = np.floor(scaled)
        elif rounding == "toward_positive":
            rounded = np.where(x < 0, np.floor(scaled), np.ceil(scaled))
        elif rounding == "toward_negative":
            rounded = np.where(x < 0, np.ceil(scaled), np.floor(scaled))
        else:
            lower = np.floor(scaled)
            delta = scaled - lower
            if rounding == "to_odd":
                odd = (np.remainder(lower, 2) != 0) if fmt.precision > 1 else ((lower != 0) & (((shift + fmt.bias) & 1) != 0))
                away = (delta > 0) & ~odd
            elif rounding == "stochastic_a":
                away = np.floor(np.ldexp(delta, srnumbits)) + bits >= 2**srnumbits
            elif rounding == "stochastic_b":
                away = np.floor(np.ldexp(delta, srnumbits + 1)) + (2.0 * bits + 1) >= 2**(srnumbits + 1)
            else:
                away = np.rint(np.ldexp(delta, srnumbits)) + bits >= 2**srnumbits
            rounded = lower + away
        result = np.asarray(np.ldexp(rounded, shift))
    maximum = fmt.max
    overflow = result > maximum
    if saturation != "none":
        preserve_inf = np.isinf(x) & (saturation == "propagate") & (fmt.domain == "extended")
        np.copyto(result, maximum, where=overflow & ~preserve_inf)
    else:
        special = np.inf if fmt.domain == "extended" else np.nan
        if rounding in ("toward_zero", "toward_positive", "toward_negative"):
            inward = np.isfinite(x)
            if rounding == "toward_positive":
                inward = inward & (x < 0)
            elif rounding == "toward_negative":
                inward = inward & (x >= 0)
            np.copyto(result, maximum, where=overflow & inward)
            overflow = overflow & ~inward
        np.copyto(result, special, where=overflow)
    np.copysign(result, x, out=result)
    if not fmt.signed:
        negative = (x < 0) & (result != 0)
        inward = (rounding in ("toward_zero", "toward_positive")) & np.isfinite(x)
        np.copyto(result, 0.0 if saturation != "none" else np.nan, where=negative)
        if saturation == "none":
            np.copyto(result, 0.0, where=negative & inward)
    np.copyto(result, 0.0, where=result == 0)
    return result


def p3109_encode(values, fmt=P3109Format(), *, rounding="nearest_even", saturate=False,
                 srbits=None, srnumbits=0, check=True):
    """Quantize and encode code points on the input backend (not bit packed).

    NumPy uses uint8/16/32; tensor backends use native signed integer codes.
    """
    fmt = _format(fmt)
    if type(check) is not bool:
        raise TypeError("check must be a bool")
    from ._p3109 import backend_of
    tensor_backend = backend_of(values)
    if tensor_backend is not None:
        if rounding not in _MODES or type(check) is not bool:
            raise ValueError("invalid rounding or check policy")
        from ._p3109.kernel import encode
        return encode(values, fmt, rounding, _saturation(saturate), srbits, srnumbits, check, tensor_backend)
    q = p3109_quantize(values, fmt, rounding=rounding, saturate=saturate,
                       srbits=srbits, srnumbits=srnumbits)
    a = np.abs(q)
    safe = np.where(np.isfinite(a), a, 0.0)
    t = fmt.precision - 1
    e = np.maximum(np.frexp(safe)[1] - 1 + fmt.bias, 0)
    fraction = np.ldexp(safe, fmt.bias + t - np.maximum(e, 1)).astype(np.uint64)
    code = (e.astype(np.uint64) << np.uint64(t)) + fraction - np.where(e > 0, 1 << t, 0).astype(np.uint64)
    if fmt.domain == "extended":
        code = np.where(np.isinf(a), fmt.max_code + 1, code).astype(np.uint64)
    if fmt.signed:
        code = code | (np.signbit(q).astype(np.uint64) << np.uint64(fmt.k - 1))
    code = np.where(np.isnan(q), fmt.nan_code, code)
    code = np.where(q == 0, 0, code)
    return code.astype(np.uint8 if fmt.k <= 8 else np.uint16 if fmt.k <= 16 else np.uint32)


def p3109_decode(codes, fmt=P3109Format(), *, check=True, dtype=None):
    """Decode validated integer codes on their native backend.

    dtype may be 'float32' or 'float64'. The default is float64 except for JAX
    when x64 is disabled. Compiled tensor callers may use check=False after
    eager validation; invalid unchecked tensor code elements become NaN.
    """
    fmt = _format(fmt)
    if type(check) is not bool:
        raise TypeError("check must be a bool")
    from ._p3109 import backend_of
    tensor_backend = backend_of(codes)
    if tensor_backend is not None:
        from ._p3109.kernel import decode
        return decode(codes, fmt, check, tensor_backend, dtype)
    if dtype not in (None, "float32", "float64"):
        raise ValueError("dtype must be None, 'float32' or 'float64'")
    if dtype == "float32" and (fmt.precision > 24 or 2 - fmt.bias - fmt.precision < -125
                              or math.frexp(fmt.max)[1] - 1 > 126):
        raise ValueError("format exceeds float32 emulation range; use float64")
    raw = np.asarray(codes)
    if raw.dtype.kind not in "iu":
        raise TypeError("codes must contain integers")
    if np.any(raw < 0) or np.any(raw >= 1 << fmt.k):
        raise ValueError(f"codes must be in [0, {1 << fmt.k})")
    code = raw.astype(np.uint64, copy=False)
    magnitude = code & np.uint64((1 << (fmt.k - int(fmt.signed))) - 1)
    t = fmt.precision - 1
    exponent = (magnitude >> np.uint64(t)).astype(np.int32)
    fraction = (magnitude & np.uint64((1 << t) - 1)).astype(np.float64)
    result = np.ldexp(fraction + np.where(exponent > 0, float(1 << t), 0.0),
                      np.maximum(exponent, 1) - fmt.bias - t)
    if fmt.domain == "extended":
        result = np.where(magnitude == fmt.max_code + 1, np.inf, result)
    if fmt.signed:
        result = np.where(code >= 1 << (fmt.k - 1), -result, result)
    result = np.where(code == fmt.nan_code, np.nan, result)
    return result.astype(np.float32) if dtype == "float32" else result


@dataclass(frozen=True)
class P3109:
    """Reusable, immutable quantizer with explicit format and rounding policy."""
    format: P3109Format = P3109Format()
    rounding: str = "nearest_even"
    saturate: object = False

    def __post_init__(self):
        _format(self.format)
        if self.rounding not in _MODES:
            raise ValueError(f"rounding must be one of {_MODES}")
        _saturation(self.saturate)

    def __call__(self, values, *, srbits=None, srnumbits=0, ste=False, check=True):
        return p3109_quantize(values, self.format, rounding=self.rounding, saturate=self.saturate,
                               srbits=srbits, srnumbits=srnumbits, ste=ste, check=check)

    def encode(self, values, *, srbits=None, srnumbits=0, check=True):
        return p3109_encode(values, self.format, rounding=self.rounding, saturate=self.saturate,
                             srbits=srbits, srnumbits=srnumbits, check=check)

    def decode(self, codes, *, check=True, dtype=None):
        return p3109_decode(codes, self.format, check=check, dtype=dtype)

    def to_dict(self):
        """Export the complete policy, including a versioned schema."""
        return {"schema_version": 1, "format": self.format.to_dict(),
                "rounding": self.rounding, "saturate": self.saturate}

    @classmethod
    def from_dict(cls, parameters):
        params = dict(parameters)
        if set(params) != {"schema_version", "format", "rounding", "saturate"}:
            raise ValueError("invalid P3109 policy fields")
        schema_version = params.pop("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("unsupported P3109 schema_version")
        params["format"] = P3109Format.from_dict(params["format"])
        return cls(**params)
