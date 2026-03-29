import numpy as np
import dask.array as da

class LightChop_:
    """
    A class to simulate different floating-point precisions and rounding modes
    for PyTorch tensors. This code implements a custom floating-point precision simulator
    that mimics IEEE 754 floating-point representation with configurable exponent bits (exp_bits),
    significand bits (sig_bits), and various rounding modes (rmode). 
    It uses PyTorch tensors for efficient computation and handles special cases like zeros,
    infinities, NaNs, and subnormal numbers. The code follows IEEE 754 conventions for sign, 
    exponent bias, implicit leading 1 (for normal numbers), and subnormal number handling.

    Initialize with specific format parameters.
    Convert to custom float representation with proper IEEE 754 handling
    
    Parameters
    ----------
    exp_bits: int 
        Number of bits for exponent.

    sig_bits : int
        Number of bits for significand (significant digits)

    rmode : int
        rounding modes.

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

    subnormal : boolean, default=True
        Whether or not to support subnormal numbers.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    chunk_size : int, default=800
        the number of elements in each smaller sub-array (or chunk) that a 
        large array is divided into for parallel processing; smaller chunks
        enable more parallelism but increase overhead, while larger chunks 
        reduce overhead but demand more memory. Essentially, chunk size is 
        the granular unit of work Dask manages, balancing 
        computation efficiency and memory constraints. 

    random_state : int, default=42
        random seed for stochastic rounding.
    """
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, 
                 subnormal: bool = True, chunk_size: int = 800, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.exp_max = 2 ** self.exp_bits - 1

        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.subnormal = subnormal
        self.sig_steps = 2 ** sig_bits
        self.chunk_size = chunk_size
        np.random.seed(random_state)

    def _to_custom_float(self, x: np.ndarray) -> tuple:
        sign = np.sign(x)
        abs_x = np.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = np.isinf(x)
        nan_mask = np.isnan(x)
        
        exponent = np.floor(np.log2(np.maximum(abs_x, 1e-38)))
        significand = abs_x / (2.0 ** exponent)
        
        if self.subnormal:
            subnormal_mask = (exponent < self.min_exp)
            significand = np.where(subnormal_mask, abs_x / (2.0 ** self.min_exp), significand)
            exponent = np.where(subnormal_mask, self.min_exp, exponent)
        else:
            subnormal_mask = (exponent < self.min_exp)
            significand = np.where(subnormal_mask, 0.0, significand)
            exponent = np.where(subnormal_mask, 0, exponent)
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask
    

    def _quantize_components(self, x: np.ndarray, sign: np.ndarray, exponent: np.ndarray, 
                            significand: np.ndarray, zero_mask: np.ndarray, 
                            inf_mask: np.ndarray, nan_mask: np.ndarray, rmode) -> np.ndarray:
        
        exponent = np.clip(exponent, 0, self.exp_max)
        
        normal_mask = (exponent > 0) & (exponent < self.exp_max)
        subnormal_mask = (exponent == 0) & (significand > 0) if self.subnormal else np.zeros_like(x, dtype=bool)
        sig_normal = significand - 1.0
        
        sig_steps = self.sig_steps
        sig_scaled = sig_normal * sig_steps
        sig_sub_scaled = significand * sig_steps if self.subnormal else None
        
        if rmode == 1:  # Nearest
            sig_q = np.round(sig_scaled) / sig_steps
            if self.subnormal:
                sig_q = np.where(subnormal_mask, np.round(sig_sub_scaled) / sig_steps, sig_q)
                
        elif rmode == 2:  # Plus infinity
            sig_q = np.where(sign > 0, np.ceil(sig_scaled), np.floor(sig_scaled)) / sig_steps
            if self.subnormal:
                sig_q = np.where(subnormal_mask, 
                            np.where(sign > 0, np.ceil(sig_sub_scaled), np.floor(sig_sub_scaled)) / sig_steps, 
                            sig_q)
                
        elif rmode == 3:  # Minus infinity
            sig_q = np.where(sign > 0, np.floor(sig_scaled), np.ceil(sig_scaled)) / sig_steps
            if self.subnormal:
                sig_q = np.where(subnormal_mask, 
                            np.where(sign > 0, np.floor(sig_sub_scaled), np.ceil(sig_sub_scaled)) / sig_steps, 
                            sig_q)
                
        elif rmode == 4:  # Towards zero
            sig_q = np.floor(sig_scaled) / sig_steps
            if self.subnormal:
                sig_q = np.where(subnormal_mask, np.floor(sig_sub_scaled) / sig_steps, sig_q)
                
        elif rmode == 5:  # Stochastic proportional
            floor_val = np.floor(sig_scaled)
            fraction = sig_scaled - floor_val
            prob = np.random.random(x.shape)
            sig_q = np.where(prob < fraction, floor_val + 1, floor_val) / sig_steps
            if self.subnormal:
                sub_floor = np.floor(sig_sub_scaled)
                sub_fraction = sig_sub_scaled - sub_floor
                sig_q = np.where(subnormal_mask, 
                            np.where(prob < sub_fraction, sub_floor + 1, sub_floor) / sig_steps, 
                            sig_q)
                
        elif rmode == 6:  # Stochastic equal
            floor_val = np.floor(sig_scaled)
            prob = np.random.random(x.shape)
            sig_q = np.where(prob < 0.5, floor_val, floor_val + 1) / sig_steps
            if self.subnormal:
                sub_floor = np.floor(sig_sub_scaled)
                sig_q = np.where(subnormal_mask, 
                            np.where(prob < 0.5, sub_floor, sub_floor + 1) / sig_steps, 
                            sig_q)
                
        elif rmode == 7:  # Nearest, ties to zero
            floor_val = np.floor(sig_scaled)
            is_half = np.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = np.where(is_half, np.where(sign >= 0, floor_val, floor_val + 1), np.round(sig_scaled)) / sig_steps
            if self.subnormal:
                sub_floor = np.floor(sig_sub_scaled)
                sub_is_half = np.abs(sig_sub_scaled - sub_floor - 0.5) < 1e-6
                sig_q = np.where(subnormal_mask, 
                            np.where(sub_is_half, np.where(sign >= 0, sub_floor, sub_floor + 1), 
                                        np.round(sig_sub_scaled)) / sig_steps, sig_q)
                
        elif rmode == 8:  # Nearest, ties away
            floor_val = np.floor(sig_scaled)
            is_half = np.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = np.where(is_half, np.where(sign >= 0, floor_val + 1, floor_val), np.round(sig_scaled)) / sig_steps
            if self.subnormal:
                sub_floor = np.floor(sig_sub_scaled)
                sub_is_half = np.abs(sig_sub_scaled - sub_floor - 0.5) < 1e-6
                sig_q = np.where(subnormal_mask, 
                            np.where(sub_is_half, np.where(sign >= 0, sub_floor + 1, sub_floor), 
                                        np.round(sig_sub_scaled)) / sig_steps, sig_q)
        
        elif rmode == 9:  # Round-to-Odd
            rounded = np.round(sig_scaled)
            sig_q = np.where(rounded % 2 == 0, 
                            rounded + np.where(sig_scaled >= rounded, 1, -1), 
                            rounded) / sig_steps
            if self.subnormal:
                sub_rounded = np.round(sig_sub_scaled)
                sig_q = np.where(subnormal_mask,
                                np.where(sub_rounded % 2 == 0,
                                        sub_rounded + np.where(sig_sub_scaled >= sub_rounded, 1, -1),
                                        sub_rounded) / sig_steps,
                                sig_q)
        
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")
        
        result = np.where(normal_mask, sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias)), 0.0)
        if self.subnormal:
            result = np.where(subnormal_mask, sign * sig_q * (2.0 ** self.min_exp), result)
        result = np.where(zero_mask, 0.0, result)

        sign = np.where(np.isclose(sign, 0), 1.0, sign)  # Replace 0 with 1 to avoid 0 * inf
        result = np.where(inf_mask, sign * np.inf, result)
        result = np.where(nan_mask, np.nan, result)
        
        return result

    def quantize(self, x: np.ndarray) -> np.ndarray:
        # Convert to Dask array if input is large
        if isinstance(x, np.ndarray) and x.size > self.chunk_size:
            x_da = da.from_array(x, chunks=self.chunk_size)
            result = x_da.map_blocks(
                lambda block: self._quantize_components(
                    block,
                    *self._to_custom_float(block),
                    self.rmode
                ),
                dtype=x.dtype
            )
            return result.compute()
        else:
            sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
            return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode)


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
        i = np.array(i, dtype=np.int32)  # Exponent not chopped
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

