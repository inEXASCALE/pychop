"""NumPy LightChop backend for fast custom floating-point quantization."""

import numpy as np
from ..cadna_random import CADNARandomGenerator
from ._light_quantize import quantize as _quantize_light


class LightChop_:
    """
    A class to simulate different floating-point precisions and rounding modes
    for NumPy arrays. This code implements a custom floating-point precision
    simulator that mimics IEEE 754 floating-point representation with configurable
    exponent bits (exp_bits), significand bits (sig_bits), and various rounding
    modes (rmode).

    Parameters
    ----------
    exp_bits : int
        Number of bits for exponent.

    sig_bits : int
        Number of bits for significand.

    rmode : int
        Rounding mode:
        - 1 : Round to nearest value, ties to even.
        - 2 : Round towards plus infinity.
        - 3 : Round towards minus infinity.
        - 4 : Truncate toward zero.
        - 5 : Stochastic rounding proportional to the fractional part.
        - 6 : Stochastic rounding with 50% probability.
        - 7 : Round to nearest value, ties to zero.
        - 8 : Round to nearest value, ties to away.
        - 9 : Round to odd.
        - 10 : CADNA-style random directed rounding.

    subnormal : bool, default=True
        Whether to support subnormal numbers. If False, subnormals are flushed to zero.

    chunk_size : int, default=800
        Accepted for compatibility; conversion uses a vectorized NumPy kernel.

    random_state : int, default=42
        Random seed for stochastic rounding. This implementation uses a per-instance
        NumPy Generator and does not modify NumPy's global RNG state.
    """

    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1,
                 subnormal: bool = True, chunk_size: int = 800, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.exp_max = 2 ** self.exp_bits - 1
        self.exp_min = 0

        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.subnormal = subnormal
        self.sig_steps = 2 ** sig_bits
        self.inv_sig_steps = 1.0 / self.sig_steps
        self.min_exp_power = 2.0 ** self.min_exp
        self.inv_min_exp_power = 1.0 / self.min_exp_power
        self.chunk_size = chunk_size

        # Per-instance RNG for stochastic rounding modes.
        # Do not call np.random.seed(...), because that mutates the global RNG state.
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        if rmode == 10:
            self._cadna_gen = CADNARandomGenerator(seed=random_state, backend="numpy")
        else:
            self._cadna_gen = None

    def _rand_like(self, x) -> np.ndarray:
        """Per-instance replacement for np.random.random used by stochastic rounding."""
        dtype = getattr(x, "dtype", np.float64)
        dtype = np.float32 if dtype == np.float32 else np.float64
        try:
            return self._rng.random(np.shape(x), dtype=dtype)
        except TypeError:
            # Older NumPy versions do not support the dtype argument here.
            return self._rng.random(np.shape(x)).astype(dtype, copy=False)

    def _to_custom_float(self, x: np.ndarray) -> tuple:
        sign = np.sign(x)
        abs_x = np.abs(x)

        zero_mask = (abs_x == 0)
        inf_mask = np.isinf(x)
        nan_mask = np.isnan(x)

        finite_nonzero = np.isfinite(x) & (abs_x != 0)
        fraction, power = np.frexp(np.where(finite_nonzero, abs_x, 1.0))
        exponent = power - 1
        significand = 2.0 * fraction
        subnormal_mask = finite_nonzero & (exponent < self.min_exp)
        if self.subnormal:
            with np.errstate(over="ignore", under="ignore"):
                sub_significand = np.ldexp(np.where(subnormal_mask, abs_x, 0.0),
                                             -self.min_exp)
            significand = np.where(subnormal_mask, sub_significand, significand)
            exponent = np.where(subnormal_mask, self.min_exp - 1, exponent)
        else:
            significand = np.where(subnormal_mask, 0.0, significand)
            exponent = np.where(subnormal_mask, self.min_exp - 1, exponent)

        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask

    def _cadna_directed_round(self, val, sign):
        """
        CADNA-style random directed rounding for positive scaled significands.

        random bit = 0: upward rounding
            positive number -> ceil(val)
            negative number -> floor(val)

        random bit = 1: downward rounding
            positive number -> floor(val)
            negative number -> ceil(val)
        """
        if self._cadna_gen is None:
            self._cadna_gen = CADNARandomGenerator(seed=self.random_state, backend="numpy")

        bits = self._cadna_gen.random_bits(np.shape(val)).astype(bool)

        floor_val = np.floor(val)
        ceil_val = np.ceil(val)

        # bit=0: upward   => sign>0 -> ceil,  sign<=0 -> floor
        # bit=1: downward => sign>0 -> floor, sign<=0 -> ceil
        choose_floor = bits == (sign > 0)

        return np.where(choose_floor, floor_val, ceil_val)

    def _quantize_components(self, x, sign, exponent, significand,
                             zero_mask, inf_mask, nan_mask, rmode):
        """Compatibility entry point for the target-grid quantizer."""
        random_values = self._rand_like(x) if rmode in (5, 6) else None
        cadna_bits = self._cadna_gen.random_bits(np.shape(x)) if rmode == 10 else None
        return _quantize_light(x, self.exp_bits, self.sig_bits, rmode,
                               self.subnormal, random_values, cadna_bits)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        random_values = self._rand_like(x) if self.rmode in (5, 6) else None
        cadna_bits = self._cadna_gen.random_bits(x.shape) if self.rmode == 10 else None
        return _quantize_light(x, self.exp_bits, self.sig_bits, self.rmode,
                               self.subnormal, random_values, cadna_bits)

    # Trigonometric Functions
    def sin(self, x):
        x = self.quantize(x)
        result = np.sin(x)
        return self.quantize(result)

    def cos(self, x):
        x = self.quantize(x)
        result = np.cos(x)
        return self.quantize(result)

    def tan(self, x):
        x = self.quantize(x)
        result = np.tan(x)
        return self.quantize(result)

    def arcsin(self, x):
        x = self.quantize(x)
        if not np.all(np.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        result = np.arcsin(x)
        return self.quantize(result)

    def arccos(self, x):
        x = self.quantize(x)
        if not np.all(np.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        result = np.arccos(x)
        return self.quantize(result)

    def arctan(self, x):
        x = self.quantize(x)
        result = np.arctan(x)
        return self.quantize(result)

    # Hyperbolic Functions
    def sinh(self, x):
        x = self.quantize(x)
        result = np.sinh(x)
        return self.quantize(result)

    def cosh(self, x):
        x = self.quantize(x)
        result = np.cosh(x)
        return self.quantize(result)

    def tanh(self, x):
        x = self.quantize(x)
        result = np.tanh(x)
        return self.quantize(result)

    def arcsinh(self, x):
        x = self.quantize(x)
        result = np.arcsinh(x)
        return self.quantize(result)

    def arccosh(self, x):
        x = self.quantize(x)
        if not np.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        result = np.arccosh(x)
        return self.quantize(result)

    def arctanh(self, x):
        x = self.quantize(x)
        if not np.all(np.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        result = np.arctanh(x)
        return self.quantize(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        x = self.quantize(x)
        result = np.exp(x)
        return self.quantize(result)

    def expm1(self, x):
        x = self.quantize(x)
        result = np.expm1(x)
        return self.quantize(result)

    def log(self, x):
        x = self.quantize(x)
        if not np.all(x > 0):
            raise ValueError("log input must be positive")
        result = np.log(x)
        return self.quantize(result)

    def log10(self, x):
        x = self.quantize(x)
        if not np.all(x > 0):
            raise ValueError("log10 input must be positive")
        result = np.log10(x)
        return self.quantize(result)

    def log2(self, x):
        x = self.quantize(x)
        if not np.all(x > 0):
            raise ValueError("log2 input must be positive")
        result = np.log2(x)
        return self.quantize(result)

    def log1p(self, x):
        x = self.quantize(x)
        if not np.all(x > -1):
            raise ValueError("log1p input must be > -1")
        result = np.log1p(x)
        return self.quantize(result)

    # Power and Root Functions
    def sqrt(self, x):
        x = self.quantize(x)
        if not np.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        result = np.sqrt(x)
        return self.quantize(result)

    def cbrt(self, x):
        x = self.quantize(x)
        result = np.cbrt(x)
        return self.quantize(result)

    # Miscellaneous Functions
    def abs(self, x):
        x = self.quantize(x)
        result = np.abs(x)
        return self.quantize(result)

    def reciprocal(self, x):
        x = self.quantize(x)
        if not np.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        result = np.reciprocal(x)
        return self.quantize(result)

    def square(self, x):
        x = self.quantize(x)
        result = np.square(x)
        return self.quantize(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        x = self.quantize(x)
        mantissa, exponent = np.frexp(x)
        return self.quantize(mantissa), self.quantize(exponent)

    def hypot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.hypot(x, y)
        return self.quantize(result)

    def diff(self, x, n=1):
        x = self.quantize(x)
        result = np.diff(x, n=n)
        return self.quantize(result)

    def power(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.power(x, y)
        return self.quantize(result)

    def modf(self, x):
        x = self.quantize(x)
        fractional, integer = np.modf(x)
        return self.quantize(fractional), self.quantize(integer)

    def ldexp(self, x, i):
        i = np.array(i, dtype=np.int32)  # Exponent not chopped.
        x = self.quantize(x)
        result = np.ldexp(x, i)
        return self.quantize(result)

    def angle(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.quantize(x)
        result = np.angle(x)
        return self.quantize(result)

    def real(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.quantize(x)
        result = np.real(x)
        return self.quantize(result)

    def imag(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.quantize(x)
        result = np.imag(x)
        return self.quantize(result)

    def conj(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.quantize(x)
        result = np.conj(x)
        return self.quantize(result)

    def maximum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.maximum(x, y)
        return self.quantize(result)

    def minimum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.minimum(x, y)
        return self.quantize(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.multiply(x, y)
        return self.quantize(result)

    def mod(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not np.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        result = np.mod(x, y)
        return self.quantize(result)

    def divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not np.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        result = np.divide(x, y)
        return self.quantize(result)

    def add(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.add(x, y)
        return self.quantize(result)

    def subtract(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.subtract(x, y)
        return self.quantize(result)

    def floor_divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not np.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        result = np.floor_divide(x, y)
        return self.quantize(result)

    def bitwise_and(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.bitwise_and(x, y)
        return self.quantize(result)

    def bitwise_or(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.bitwise_or(x, y)
        return self.quantize(result)

    def bitwise_xor(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.bitwise_xor(x, y)
        return self.quantize(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        x = self.quantize(x)
        result = np.sum(x, axis=axis)
        return self.quantize(result)

    def prod(self, x, axis=None):
        x = self.quantize(x)
        result = np.prod(x, axis=axis)
        return self.quantize(result)

    def mean(self, x, axis=None):
        x = self.quantize(x)
        result = np.mean(x, axis=axis)
        return self.quantize(result)

    def std(self, x, axis=None):
        x = self.quantize(x)
        result = np.std(x, axis=axis)
        return self.quantize(result)

    def var(self, x, axis=None):
        x = self.quantize(x)
        result = np.var(x, axis=axis)
        return self.quantize(result)

    def dot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.dot(x, y)
        return self.quantize(result)

    def matmul(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.matmul(x, y)
        return self.quantize(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        x = self.quantize(x)
        result = np.floor(x)
        return self.quantize(result)

    def ceil(self, x):
        x = self.quantize(x)
        result = np.ceil(x)
        return self.quantize(result)

    def round(self, x, decimals=0):
        x = self.quantize(x)
        result = np.round(x, decimals=decimals)
        return self.quantize(result)

    def sign(self, x):
        x = self.quantize(x)
        result = np.sign(x)
        return self.quantize(result)

    def clip(self, x, a_min, a_max):
        a_min = np.array(a_min, dtype=np.float64)
        a_max = np.array(a_max, dtype=np.float64)
        x = self.quantize(x)
        chopped_a_min = self.quantize(a_min)
        chopped_a_max = self.quantize(a_max)
        result = np.clip(x, chopped_a_min, chopped_a_max)
        return self.quantize(result)

    # Special Functions
    def erf(self, x):
        x = self.quantize(x)
        result = np.special.erf(x)
        return self.quantize(result)

    def erfc(self, x):
        x = self.quantize(x)
        result = np.special.erfc(x)
        return self.quantize(result)

    def gamma(self, x):
        x = self.quantize(x)
        result = np.special.gamma(x)
        return self.quantize(result)

    # New Mathematical Functions
    def fabs(self, x):
        """Floating-point absolute value with chopping."""
        x = self.quantize(x)
        result = np.fabs(x)
        return self.quantize(result)

    def logaddexp(self, x, y):
        """Logarithm of sum of exponentials with chopping."""
        x = self.quantize(x)
        y = self.quantize(y)
        result = np.logaddexp(x, y)
        return self.quantize(result)

    def cumsum(self, x, axis=None):
        """Cumulative sum with chopping."""
        x = self.quantize(x)
        result = np.cumsum(x, axis=axis)
        return self.quantize(result)

    def cumprod(self, x, axis=None):
        """Cumulative product with chopping."""
        x = self.quantize(x)
        result = np.cumprod(x, axis=axis)
        return self.quantize(result)

    def degrees(self, x):
        """Convert radians to degrees with chopping."""
        x = self.quantize(x)
        result = np.degrees(x)
        return self.quantize(result)

    def radians(self, x):
        """Convert degrees to radians with chopping."""
        x = self.quantize(x)
        result = np.radians(x)
        return self.quantize(result)

    def __call__(self, x: np.ndarray):
        return self.quantize(x)
