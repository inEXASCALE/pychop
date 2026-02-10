import numpy as np
from typing import Tuple


class FPRound:
    def __init__(self, ibits: int, fbits: int, rmode: int=1):
        """
        Initialize fixed-point simulator.
        
        Parameters
        ----------  
        ibits : int, default=4
            The bitwidth of integer part. 
    
        fbits : int, default=4
            The bitwidth of fractional part. 

        rmode : str | int
            - 0 or "nearest_odd": Round to nearest value, ties to odd (Not implemented). 
            - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
            - 2 or "plus_inf": Round towards plus infinity (round up).
            - 3 or "minus_inf": Round towards minus infinity (round down).
            - 4 or "toward_zero": Truncate toward zero (no rounding up).
            - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
            - 6 or "stoc_equal": Stochastic rounding with 50% probability.

        """
        
        self.ibits = ibits
        self.fbits = fbits
        self.total_bits = ibits + fbits
        self.max_value = 2 ** (ibits - 1) - 2 ** (-fbits)
        self.min_value = -2 ** (ibits - 1)
        self.resolution = 2 ** (-fbits)
        self.rmode = rmode
        
    def _to_fixed_point_components(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract sign and magnitude from floating-point input.
        
        Parameters
        ----------  
        x : numpy.ndarray
            Input tensor
        
        Returns
        ----------  
        sign: Tensor of signs (+1 or -1)
            abs_x: Tensor of absolute values
        """
        
        sign = np.sign(x)  # 1, -1, or 0
        abs_x = np.abs(x)
        return sign, abs_x

    def _quantize(self, 
                  x: np.ndarray,
                  sign: np.ndarray,
                  abs_x: np.ndarray) -> np.ndarray:
        """
        Quantize to fixed-point with specified rounding mode.
        
        Parameters
        ----------  
        x : numpy.ndarray
            Input tensor
            
        sign : numpy.ndarray
            Signs of input values
            
        abs_x : numpy.ndarray
            Absolute values of input
            
        Returns
        ----------  
        result : numpy.ndarray
            Quantized tensor in fixed-point representation
        """
        


        scaled = abs_x / self.resolution

        if self.rmode in {"nearest", 1}:
            quantized = np.round(scaled)

        elif self.rmode in {"plus_inf", 2}:
            quantized = np.where(sign > 0, np.ceil(scaled), np.floor(scaled))

        elif self.rmode in {"minus_inf", 3}:
            quantized = np.where(sign > 0, np.floor(scaled), np.ceil(scaled))

        elif self.rmode in {"towards_zero", 4}:
            quantized = np.trunc(scaled)

        elif self.rmode in {"stoc_prop", 5}:
            floor_val = np.floor(scaled)
            fraction = scaled - floor_val
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < fraction, floor_val + 1, floor_val)

        elif self.rmode in {"stoc_equal", 6}:
            floor_val = np.floor(scaled)
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < 0.5, floor_val, floor_val + 1)

        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")

        result = sign * quantized * self.resolution
        result = np.clip(result, self.min_value, self.max_value)

        result[np.isinf(x)] = np.sign(x[np.isinf(x)]) * self.max_value
        result[np.isnan(x)] = np.nan

        return result

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : numpy.ndarray
            Input tensor
            
        Returns
        ----------  
        result : numpy.ndarray
            Quantized tensor in fixed-point representation
        """
        sign, abs_x = self._to_fixed_point_components(x)
        return self._quantize(x, sign, abs_x)

    
    def get_format_info(self) -> dict:
        """Return information about the fixed-point format."""
        return {
            "format": f"Q{self.ibits}.{self.fbits}",
            "total_bits": self.total_bits,
            "range": (self.min_value, self.max_value),
            "resolution": self.resolution
        }

    def __call__(self, x):
        """
        x : numpy.ndarray | jax.Array | torch.Tensor,
            The input array. 

        Returns
        ----------
        x_q : numpy.ndarray | jax.Array | torch.Tensor, 
            The quantized array.
        """
        return self.quantize(x)

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

