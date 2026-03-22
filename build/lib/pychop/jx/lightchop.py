import jax
import jax.numpy as jnp
from jax import jit

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
        self.rng_key = jax.random.PRNGKey(random_state)

        # JIT-compile the quantization core. 
        # static_argnums=(7,) corresponds to 'rmode' to allow specialized kernels.
        self._quantize_core = jit(self._quantize_impl, static_argnums=(7,))

    def _to_custom_float(self, x):
        """Decomposes x into sign, exponent, and significand (vectorized)."""
        sign = jnp.sign(x)
        # Fix sign of 0 to be positive for standard handling, though IEEE supports -0.
        # Here we just preserve the sign bit as extracted.
        
        abs_x = jnp.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = jnp.isinf(x)
        nan_mask = jnp.isnan(x)
        
        # Calculate exponent
        # Use maximum(..., 1e-38) to avoid log2(0)
        exponent = jnp.floor(jnp.log2(jnp.maximum(abs_x, 1e-45)))
        significand = abs_x / (2.0 ** exponent)
        
        if self.subnormal:
            # If exponent is below min_exp, treat as subnormal
            subnormal_mask = (exponent < self.min_exp)
            # Recalculate significand for subnormals: value / min_subnormal_scale
            significand = jnp.where(subnormal_mask, abs_x / (2.0 ** self.min_exp), significand)
            exponent = jnp.where(subnormal_mask, self.min_exp, exponent)
        else:
            # Flush subnormals to zero
            subnormal_mask = (exponent < self.min_exp)
            significand = jnp.where(subnormal_mask, 0.0, significand)
            exponent = jnp.where(subnormal_mask, 0.0, exponent) # Set to arbitrary valid exp
            zero_mask = zero_mask | subnormal_mask

        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask

    def _quantize_impl(self, x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, rmode, noise):
        """
        Core quantization logic. 
        'noise' is a pre-generated array of random values [0, 1] used for stochastic modes.
        """
        exp_max_val = 2 ** self.exp_bits - 1
        
        # Clip exponent to representable range
        exponent = jnp.clip(exponent, 0, exp_max_val)
        
        # Identify normal vs subnormal in the target format
        # Note: 'significand' coming in is in [1.0, 2.0) for normals (mostly)
        
        # For our extracted components, subnormal handling in target:
        # If we allowed subnormals in decomposition, they are already scaled relative to min_exp.
        # We just need to check if the *target* exponent is 0.
        subnormal_mask = (exponent == 0) if self.subnormal else jnp.zeros_like(x, dtype=bool)

        # Scale significand to integer range for rounding
        # Normals: Remove implicit 1.0, scale fractional part
        # Subnormals: Scale full value
        sig_steps = self.sig_steps
        
        # Prepare values for rounding
        # For normals: we round (significand - 1.0) * steps
        # For subnormals: we round significand * steps
        sig_to_round_normal = (significand - 1.0) * sig_steps
        sig_to_round_sub = significand * sig_steps
        
        # Select the value to round based on whether it's normal or subnormal
        val_to_round = jnp.where(subnormal_mask, sig_to_round_sub, sig_to_round_normal)
        
        # --- Rounding Implementations ---
        
        def nearest(val):
            return jnp.round(val)

        def plus_inf(val, sgn):
            # If positive, ceil. If negative, floor.
            return jnp.where(sgn > 0, jnp.ceil(val), jnp.floor(val))

        def minus_inf(val, sgn):
            # If positive, floor. If negative, ceil.
            return jnp.where(sgn > 0, jnp.floor(val), jnp.ceil(val))

        def towards_zero(val):
            return jnp.floor(val)

        def stoc_prop(val, noise):
            # Stochastic proportional
            floor_val = jnp.floor(val)
            fraction = val - floor_val
            # Round up if random noise < fraction
            return jnp.where(noise < fraction, floor_val + 1.0, floor_val)

        def stoc_equal(val, noise):
            # Stochastic 50/50
            floor_val = jnp.floor(val)
            # Round up if random noise < 0.5
            return jnp.where(noise < 0.5, floor_val, floor_val + 1.0)

        def nearest_ties_zero(val, sgn):
            floor_val = jnp.floor(val)
            is_half = jnp.abs(val - floor_val - 0.5) < 1e-6
            # If half: round towards zero (floor for pos, ceil for neg? No, magnitude towards zero)
            # Floor is closer to zero for positive numbers. 
            # Actually "ties to zero" usually means round 1.5 -> 1, 2.5 -> 2? No that's ties to even.
            # Ties to zero: 1.5 -> 1, -1.5 -> -1.
            round_half = jnp.where(sgn >= 0, floor_val, floor_val + 1.0)
            return jnp.where(is_half, round_half, jnp.round(val))

        def nearest_ties_away(val, sgn):
            floor_val = jnp.floor(val)
            is_half = jnp.abs(val - floor_val - 0.5) < 1e-6
            # Ties away from zero: 1.5 -> 2, -1.5 -> -2
            round_half = jnp.where(sgn >= 0, floor_val + 1.0, floor_val)
            return jnp.where(is_half, round_half, jnp.round(val))
            
        def round_to_odd(val):
            rounded = jnp.round(val)
            # If result is even, nudge it to the nearest odd integer
            # We nudge towards the original value to minimize error
            # If val > rounded (e.g. 2.1 -> 2), nudge up (+1)
            # If val < rounded (e.g. 1.9 -> 2), nudge down (-1)
            diff = val - rounded
            nudge = jnp.sign(diff)
            # If diff is exactly 0 (integer), strictly speaking we force odd. 
            # Usually RTO is for sticky bits, here we just force odd integer.
            # Let's default nudge +1 if exactly even integer.
            nudge = jnp.where(nudge == 0, 1.0, nudge) 
            
            return jnp.where(rounded % 2 == 0, rounded + nudge, rounded)

        # Dispatcher
        if rmode == 1:
            rounded_val = nearest(val_to_round)
        elif rmode == 2:
            rounded_val = plus_inf(val_to_round, sign)
        elif rmode == 3:
            rounded_val = minus_inf(val_to_round, sign)
        elif rmode == 4:
            rounded_val = towards_zero(val_to_round)
        elif rmode == 5:
            rounded_val = stoc_prop(val_to_round, noise)
        elif rmode == 6:
            rounded_val = stoc_equal(val_to_round, noise)
        elif rmode == 7:
            rounded_val = nearest_ties_zero(val_to_round, sign)
        elif rmode == 8:
            rounded_val = nearest_ties_away(val_to_round, sign)
        elif rmode == 9:
            rounded_val = round_to_odd(val_to_round)
        else:
            # Fallback to nearest
            rounded_val = nearest(val_to_round)

        # Scale back down
        sig_q = rounded_val / sig_steps
        
        # Reconstruct
        # If normal: value = (1 + sig_q) * 2^(exp - bias)
        # If subnormal: value = sig_q * 2^(min_exp)
        
        res_normal = sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias))
        res_subnormal = sign * sig_q * (2.0 ** self.min_exp)
        
        result = jnp.where(subnormal_mask, res_subnormal, res_normal)
        
        # Apply standard masks
        result = jnp.where(zero_mask, 0.0, result)
        result = jnp.where(inf_mask, sign * jnp.inf, result)
        result = jnp.where(nan_mask, jnp.nan, result)
        
        return result


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
        x = jnp.asarray(x)
        
        # Split key only once per call
        key, subkey = jax.random.split(self.rng_key)
        self.rng_key = key
        
        # Generate noise tensor for stochastic modes (unused for deterministic modes, but cheap)
        # We generate it every time to keep the API clean, but JAX's lazy evaluation 
        # might optimize it out if rmode is static and deterministic.
        if self.rmode in [5, 6]:
            noise = jax.random.uniform(subkey, shape=x.shape)
        else:
            noise = jnp.zeros_like(x) # Dummy placeholder
            
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        
        return self._quantize_core(
            x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode, noise
        )