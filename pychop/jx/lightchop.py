"""JAX LightChop backend for fast custom floating-point quantization."""

import jax
import jax.numpy as jnp
from ..cadna_random import CADNARandomGenerator

class LightChop_:
    """
    A class to simulate different floating-point precisions and rounding modes
    for JAX arrays. This code implements a custom floating-point precision simulator
    that mimics IEEE 754 floating-point representation with configurable exponent bits (exp_bits),
    significand bits (sig_bits), and various rounding modes (rmode). 
    It uses JAX arrays for efficient computation and handles special cases like zeros,
    infinities, NaNs, and subnormal numbers. The code follows IEEE 754 conventions for sign, 
    exponent bias, implicit leading 1 (for normal numbers), and subnormal number handling.

    Parameters
    ----------
    exp_bits: int 
        Number of bits for exponent.
    sig_bits : int
        Number of bits for significand (significant digits)
    rmode : int
        Rounding mode to use when quantizing the significand. Options are:
        - 1 : Round to nearest value, ties to even (IEEE 754 default).
        - 2 : Round towards plus infinity (round up).
        - 3 : Round towards minus infinity (round down).
        - 4 : Truncate toward zero (no rounding up).
        - 5 : Stochastic rounding proportional to the fractional part.
        - 6 : Stochastic rounding with 50% probability.
        - 7 : Round to nearest value, ties to zero.
        - 8 : Round to nearest value, ties to away.
        - 9 : Round to odd.
        - 10 : CADNA-style random directed rounding.

    random_state : int, default=42
        Random seed for stochastic rounding.
    
    subnormal : bool, default=True
        Whether to support subnormal numbers.
    
    chunk_size : int, default=1000
        Chunk size for processing large arrays (not used in this implementation).
    """
    
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True, 
                 chunk_size: int = 1000, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        # Precompute constants
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.subnormal = subnormal
        self.sig_steps = 2.0 ** sig_bits
        self.chunk_size = chunk_size
        # Random state management
        self.random_state = random_state
        self.rng_key = jax.random.PRNGKey(random_state)
        self._bit_quantizers = {}
        if rmode == 10:
            self._cadna_gen = CADNARandomGenerator(seed=random_state, backend="jax")
        else:
            self._cadna_gen = None

    def _to_custom_float(self, x):
        """Decompose source bits exactly, including source subnormal values."""
        x = jnp.asarray(x)
        is_float32 = x.dtype == jnp.float32
        source_sig_bits = 23 if is_float32 else 52
        source_bias = 127 if is_float32 else 1023
        source_exp_bits = 8 if is_float32 else 11
        bit_dtype = jnp.uint32 if is_float32 else jnp.uint64
        raw_bits = x.view(bit_dtype)
        sign = jnp.where((raw_bits >> (source_sig_bits + source_exp_bits)) != 0, -1.0, 1.0)
        bits = raw_bits & ((1 << (source_sig_bits + source_exp_bits)) - 1)
        raw_exp = (bits >> source_sig_bits) & ((1 << source_exp_bits) - 1)
        fraction = bits & ((1 << source_sig_bits) - 1)
        zero_mask = (raw_exp == 0) & (fraction == 0)
        inf_mask = (raw_exp == (1 << source_exp_bits) - 1) & (fraction == 0)
        nan_mask = (raw_exp == (1 << source_exp_bits) - 1) & (fraction != 0)
        active = ~(zero_mask | inf_mask | nan_mask)
        whole_sig = jnp.where(raw_exp == 0, fraction, fraction | (1 << source_sig_bits))
        source_power = jnp.where(raw_exp == 0, 1 - source_bias,
                                 raw_exp.astype(jnp.int32) - source_bias) - source_sig_bits
        _, fraction_exp = jnp.frexp(jnp.maximum(fraction, 1).astype(x.dtype))
        actual_exp = jnp.where(raw_exp == 0, fraction_exp - 1 + source_power,
                               raw_exp.astype(jnp.int32) - source_bias)
        normal_sig = jnp.ldexp(whole_sig.astype(x.dtype), source_power - actual_exp)
        sub = active & (actual_exp < self.min_exp)

        # Reconstruct source-subnormal significands from bits because JAX may
        # flush floating subnormal intermediate results on some devices.
        sub_power = source_power - self.min_exp
        source_min_sub_exp = 1 - source_bias - source_sig_bits
        bit_shift = sub_power - source_min_sub_exp
        sub_bits = jnp.where(
            bit_shift >= 0,
            whole_sig << jnp.clip(bit_shift, 0, 63).astype(bit_dtype),
            whole_sig >> jnp.clip(-bit_shift, 0, 63).astype(bit_dtype),
        )
        sub_bit_value = sub_bits.astype(bit_dtype).view(x.dtype)
        sub_float_value = jnp.ldexp(whole_sig.astype(x.dtype), sub_power)
        sub_sig = jnp.where(actual_exp - self.min_exp < 1 - source_bias,
                            sub_bit_value, sub_float_value)
        significand = jnp.where(sub, sub_sig if self.subnormal else 0.0, normal_sig)
        exponent = jnp.where(sub, self.min_exp - 1, actual_exp)
        if not self.subnormal:
            zero_mask = zero_mask | sub
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask

    def _quantize_impl(self, x, sign, exponent, significand, zero_mask,
                       inf_mask, nan_mask, rmode, noise):
        """Compatibility entry point; conversion uses the native bit quantizer."""
        if rmode != self.rmode:
            raise ValueError("rmode does not match this LightChop instance")
        return self.__call__(x)

    # Trigonometric Functions
    def sin(self, x):
        
        x = self.quantize(x)
        
        result = jnp.sin(x)
        return self.quantize(result)

    def cos(self, x):
        
        x = self.quantize(x)
        
        result = jnp.cos(x)
        return self.quantize(result)

    def tan(self, x):
        
        x = self.quantize(x)
        
        result = jnp.tan(x)
        return self.quantize(result)

    def arcsin(self, x):
        
        x = self.quantize(x)
        if not jnp.all(jnp.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        
        result = jnp.arcsin(x)
        return self.quantize(result)

    def arccos(self, x):
        
        x = self.quantize(x)
        if not jnp.all(jnp.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        
        result = jnp.arccos(x)
        return self.quantize(result)

    def arctan(self, x):
        
        x = self.quantize(x)
        
        result = jnp.arctan(x)
        return self.quantize(result)

    # Hyperbolic Functions
    def sinh(self, x):
        
        x = self.quantize(x)
        
        result = jnp.sinh(x)
        return self.quantize(result)

    def cosh(self, x):
        
        x = self.quantize(x)
        
        result = jnp.cosh(x)
        return self.quantize(result)

    def tanh(self, x):
        
        x = self.quantize(x)
        
        result = jnp.tanh(x)
        return self.quantize(result)

    def arcsinh(self, x):
        
        x = self.quantize(x)
        
        result = jnp.arcsinh(x)
        return self.quantize(result)

    def arccosh(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        
        result = jnp.arccosh(x)
        return self.quantize(result)

    def arctanh(self, x):
        
        x = self.quantize(x)
        if not jnp.all(jnp.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        
        result = jnp.arctanh(x)
        return self.quantize(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        
        x = self.quantize(x)
        
        result = jnp.exp(x)
        return self.quantize(result)

    def expm1(self, x):
        
        x = self.quantize(x)
        
        result = jnp.expm1(x)
        return self.quantize(result)

    def log(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x > 0):
            raise ValueError("log input must be positive")
        
        result = jnp.log(x)
        return self.quantize(result)

    def log10(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x > 0):
            raise ValueError("log10 input must be positive")
        
        result = jnp.log10(x)
        return self.quantize(result)

    def log2(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x > 0):
            raise ValueError("log2 input must be positive")
        
        result = jnp.log2(x)
        return self.quantize(result)

    def log1p(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x > -1):
            raise ValueError("log1p input must be > -1")
        
        result = jnp.log1p(x)
        return self.quantize(result)

    # Power and Root Functions
    def sqrt(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        
        result = jnp.sqrt(x)
        return self.quantize(result)

    def cbrt(self, x):
        
        x = self.quantize(x)
        
        result = jnp.cbrt(x)
        return self.quantize(result)

    # Miscellaneous Functions
    def abs(self, x):
        
        x = self.quantize(x)
        
        result = jnp.abs(x)
        return self.quantize(result)

    def reciprocal(self, x):
        
        x = self.quantize(x)
        if not jnp.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        
        result = jnp.reciprocal(x)
        return self.quantize(result)

    def square(self, x):
        
        x = self.quantize(x)
        
        result = jnp.square(x)
        return self.quantize(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        
        x = self.quantize(x)
        mantissa, exponent = jnp.frexp(x)
        
        return self.quantize(mantissa), exponent  # Exponent typically not chopped

    def hypot(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.hypot(x, y)
        return self.quantize(result)

    def diff(self, x, n=1):
        
        x = self.quantize(x)
        for _ in range(n):
            x = jnp.diff(x)
        
        return self.quantize(x)

    def power(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.power(x, y)
        return self.quantize(result)

    def modf(self, x):
        
        x = self.quantize(x)
        fractional, integer = jnp.modf(x)
        
        fractional = self.quantize(fractional)
        
        integer = self.quantize(integer)
        return fractional, integer

    def ldexp(self, x, i):
        
        x = self.quantize(x)
        i = jnp.array(i, dtype=jnp.int32)  # Exponent not chopped
        
        result = jnp.ldexp(x, i)
        return self.quantize(result)

    def angle(self, x):
        
        x = self.quantize(x)
        
        result = jnp.angle(x) if jnp.iscomplexobj(x) else jnp.arctan2(x, jnp.zeros_like(x))
        return self.quantize(result)

    def real(self, x):
        
        x = self.quantize(x)
        
        result = jnp.real(x) if jnp.iscomplexobj(x) else x
        return self.quantize(result)

    def imag(self, x):
        
        x = self.quantize(x)
        
        result = jnp.imag(x) if jnp.iscomplexobj(x) else jnp.zeros_like(x)
        return self.quantize(result)

    def conj(self, x):
        
        x = self.quantize(x)
        
        result = jnp.conj(x) if jnp.iscomplexobj(x) else x
        return self.quantize(result)

    def maximum(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.maximum(x, y)
        return self.quantize(result)

    def minimum(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.minimum(x, y)
        return self.quantize(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.multiply(x, y)
        return self.quantize(result)

    def mod(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        if not jnp.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        
        result = jnp.mod(x, y)
        return self.quantize(result)

    def divide(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        if not jnp.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        
        result = jnp.divide(x, y)
        return self.quantize(result)

    def add(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.add(x, y)
        return self.quantize(result)

    def subtract(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.subtract(x, y)
        return self.quantize(result)

    def floor_divide(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        if not jnp.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        
        result = jnp.floor_divide(x, y)
        return self.quantize(result)

    def bitwise_and(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.bitwise_and(x.astype(jnp.int32), y.astype(jnp.int32)).astype(jnp.float32)
        return self.quantize(result)

    def bitwise_or(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.bitwise_or(x.astype(jnp.int32), y.astype(jnp.int32)).astype(jnp.float32)
        return self.quantize(result)

    def bitwise_xor(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.bitwise_xor(x.astype(jnp.int32), y.astype(jnp.int32)).astype(jnp.float32)
        return self.quantize(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.sum(x, axis=axis)
        return self.quantize(result)

    def prod(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.prod(x, axis=axis)
        return self.quantize(result)

    def mean(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.mean(x, axis=axis)
        return self.quantize(result)

    def std(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.std(x, axis=axis)
        return self.quantize(result)

    def var(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.var(x, axis=axis)
        return self.quantize(result)

    def dot(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.dot(x, y)
        return self.quantize(result)

    def matmul(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jnp.matmul(x, y)
        return self.quantize(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        
        x = self.quantize(x)
        
        result = jnp.floor(x)
        return self.quantize(result)

    def ceil(self, x):
        
        x = self.quantize(x)
        
        result = jnp.ceil(x)
        return self.quantize(result)

    def round(self, x, decimals=0):
        
        x = self.quantize(x)
        if decimals == 0:
            result = jnp.round(x)
        else:
            factor = 10 ** decimals
            result = jnp.round(x * factor) / factor
        
        return self.quantize(result)

    def sign(self, x):
        
        x = self.quantize(x)
        
        result = jnp.sign(x)
        return self.quantize(result)

    def clip(self, x, a_min, a_max):
        
        x = self.quantize(x)
        a_min = jnp.array(a_min, dtype=jnp.float32)
        a_max = jnp.array(a_max, dtype=jnp.float32)
        
        chopped_a_min = self.quantize(a_min)
        
        chopped_a_max = self.quantize(a_max)
        
        result = jnp.clip(x, chopped_a_min, chopped_a_max)
        return self.quantize(result)

    # Special Functions
    def erf(self, x):
        
        x = self.quantize(x)
        
        result = jax.scipy.special.erf(x)
        return self.quantize(result)

    def erfc(self, x):
        
        x = self.quantize(x)
        
        result = jax.scipy.special.erfc(x)
        return self.quantize(result)

    def gamma(self, x):
        
        x = self.quantize(x)
        
        result = jax.scipy.special.gamma(x)
        return self.quantize(result)

    # New Mathematical Functions
    def fabs(self, x):
        
        x = self.quantize(x)
        
        result = jnp.fabs(x)
        return self.quantize(result)

    def logaddexp(self, x, y):
        
        x = self.quantize(x)
        
        y = self.quantize(y)
        
        result = jax.scipy.special.logsumexp(jnp.stack([x, y]), axis=0)
        return self.quantize(result)

    def cumsum(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.cumsum(x, axis=axis)
        return self.quantize(result)

    def cumprod(self, x, axis=None):
        
        x = self.quantize(x)
        
        result = jnp.cumprod(x, axis=axis)
        return self.quantize(result)

    def degrees(self, x):
        
        x = self.quantize(x)
        
        result = jnp.degrees(x)
        return self.quantize(result)

    def radians(self, x):
        
        x = self.quantize(x)
        
        result = jnp.radians(x)
        return self.quantize(result)
        
    def __call__(self, x):
        """
        Main entry point. Quantizes x.
        """
        from .bitchop import Bitchop

        x = jnp.asarray(x)
        source_dtype = x.dtype
        if x.dtype not in (jnp.float32, jnp.float64):
            x = x.astype(jnp.float32)
        if x.dtype == jnp.float32 and self.exp_bits > 8 and self.sig_bits >= 23:
            # Every float32 value is already exact in this wider target format.
            return x.astype(source_dtype) if source_dtype in (jnp.float16, jnp.bfloat16) else x
        # JIT tracers have no concrete device until the compiled call runs.
        device = getattr(x, "device", None)
        quantizer = self._bit_quantizers.get(device)
        if quantizer is None:
            quantizer = Bitchop(
                self.exp_bits, self.sig_bits, rmode=self.rmode,
                subnormal=self.subnormal, random_state=self.random_state,
                device=device, allow_wider_source_range=True,
            )
            if device is not None:
                self._bit_quantizers[device] = quantizer
        result = quantizer(x)
        if source_dtype in (jnp.float16, jnp.bfloat16, jnp.float32):
            return result.astype(source_dtype)
        return result

    def quantize(self, x):
        return self.__call__(x)
