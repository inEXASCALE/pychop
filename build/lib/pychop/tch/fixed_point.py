import torch
from typing import Tuple

class FPRound:
    def __init__(self, ibits: int=4, fbits: int=4, rmode: int=1):
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

    def _to_fixed_point_components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract sign and magnitude from floating-point input.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
        
        Returns
        ----------  
        sign: Tensor of signs (+1 or -1)
        abs_x: Tensor of absolute values
        """
        sign = torch.sign(x)  # 1, -1, or 0
        abs_x = torch.abs(x)
        return sign, abs_x

    def _quantize(self, 
                  x: torch.Tensor,
                  sign: torch.Tensor,
                  abs_x: torch.Tensor) -> torch.Tensor:
        """
        Quantize to fixed-point with specified rounding mode and STE.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor (unquantized, for STE)
            
        sign : torch.Tensor
            Signs of input values
            
        abs_x : torch.Tensor
            Absolute values of input
            
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation with STE applied
        """
        scaled = abs_x / self.resolution

        if self.rmode in {"nearest", 1}:
            quantized = torch.round(scaled)
        elif self.rmode in {"plus_inf", 2}:
            quantized = torch.where(sign > 0, torch.ceil(scaled), torch.floor(scaled))
        elif self.rmode in {"minus_inf", 3}:
            quantized = torch.where(sign > 0, torch.floor(scaled), torch.ceil(scaled))
        elif self.rmode in {"towards_zero", 4}:
            quantized = torch.trunc(scaled)
        elif self.rmode in {"stoc_prop", 5}:
            floor_val = torch.floor(scaled)
            fraction = scaled - floor_val
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < fraction, floor_val + 1, floor_val)
        elif self.rmode in {"stoc_equal", 6}:
            floor_val = torch.floor(scaled)
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < 0.5, floor_val, floor_val + 1)
        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")

        # Compute quantized result in floating-point domain
        result = sign * quantized * self.resolution
        result = torch.clamp(result, self.min_value, self.max_value)

        # Handle infinities and NaNs
        result[torch.isinf(x)] = torch.sign(x[torch.isinf(x)]) * self.max_value
        result[torch.isnan(x)] = float('nan')

        # Apply Straight-Through Estimator (STE) if gradients are needed
        if x.requires_grad:
            result = x + (result - x).detach()

        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method and STE.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
                        
        Returns
        ----------  
        result : torch.Tensor
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
        result = torch.sin(x)
        return self.quantize(result)

    def cos(self, x):
        x = self.quantize(x)
        result = torch.cos(x)
        return self.quantize(result)

    def tan(self, x):
        x = self.quantize(x)
        result = torch.tan(x)
        return self.quantize(result)

    def arcsin(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        result = torch.asin(x)
        return self.quantize(result)

    def arccos(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        result = torch.acos(x)
        return self.quantize(result)

    def arctan(self, x):
        x = self.quantize(x)
        result = torch.atan(x)
        return self.quantize(result)

    # Hyperbolic Functions
    def sinh(self, x):
        x = self.quantize(x)
        result = torch.sinh(x)
        return self.quantize(result)

    def cosh(self, x):
        x = self.quantize(x)
        result = torch.cosh(x)
        return self.quantize(result)

    def tanh(self, x):
        x = self.quantize(x)
        result = torch.tanh(x)
        return self.quantize(result)

    def arcsinh(self, x):
        x = self.quantize(x)
        result = torch.asinh(x)
        return self.quantize(result)

    def arccosh(self, x):
        x = self.quantize(x)
        if not torch.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        result = torch.acosh(x)
        return self.quantize(result)

    def arctanh(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        result = torch.atanh(x)
        return self.quantize(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        x = self.quantize(x)
        result = torch.exp(x)
        return self.quantize(result)

    def expm1(self, x):
        x = self.quantize(x)
        result = torch.expm1(x)
        return self.quantize(result)

    def log(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log input must be positive")
        result = torch.log(x)
        return self.quantize(result)

    def log10(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log10 input must be positive")
        result = torch.log10(x)
        return self.quantize(result)

    def log2(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log2 input must be positive")
        result = torch.log2(x)
        return self.quantize(result)

    def log1p(self, x):
        x = self.quantize(x)
        if not torch.all(x > -1):
            raise ValueError("log1p input must be > -1")
        result = torch.log1p(x)
        return self.quantize(result)

    # Power and Root Functions
    def sqrt(self, x):
        x = self.quantize(x)
        if not torch.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        result = torch.sqrt(x)
        return self.quantize(result)

    def cbrt(self, x):
        x = self.quantize(x)
        result = torch.pow(x, 1/3)
        return self.quantize(result)

    # Miscellaneous Functions
    def abs(self, x):
        x = self.quantize(x)
        result = torch.abs(x)
        return self.quantize(result)

    def reciprocal(self, x):
        x = self.quantize(x)
        if not torch.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        result = torch.reciprocal(x)
        return self.quantize(result)

    def square(self, x):
        x = self.quantize(x)
        result = torch.square(x)
        return self.quantize(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        x = self.quantize(x)
        mantissa, exponent = torch.frexp(x)
        return self.quantize(mantissa), exponent  # Exponent typically not chopped

    def hypot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.hypot(x, y)
        return self.quantize(result)

    def diff(self, x, n=1):
        x = self.quantize(x)
        for _ in range(n):
            x = torch.diff(x)
        return self.quantize(x)

    def power(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.pow(x, y)
        return self.quantize(result)

    def modf(self, x):
        x = self.quantize(x)
        fractional, integer = torch.modf(x)
        return self.quantize(fractional), self.quantize(integer)

    def ldexp(self, x, i):
        x = self.quantize(x)
        i = torch.tensor(i, dtype=torch.int32, device=x.device)  # Exponent not chopped
        result = x * torch.pow(2.0, i)
        return self.quantize(result)

    def angle(self, x):
        if torch.is_complex(x):
            x = self.quantize(x)
            result = torch.angle(x)
        else:
            x = self.quantize(x)
            result = torch.atan2(x, torch.zeros_like(x))
        return self.quantize(result)

    def real(self, x):
        x = self.quantize(x)
        result = torch.real(x) if torch.is_complex(x) else x
        return self.quantize(result)

    def imag(self, x):
        x = self.quantize(x)
        result = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(x)
        return self.quantize(result)

    def conj(self, x):
        x = self.quantize(x)
        result = torch.conj(x) if torch.is_complex(x) else x
        return self.quantize(result)

    def maximum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.maximum(x, y)
        return self.quantize(result)

    def minimum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.minimum(x, y)
        return self.quantize(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.mul(x, y)
        return self.quantize(result)

    def mod(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        result = torch.fmod(x, y)
        return self.quantize(result)

    def divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        result = torch.div(x, y)
        return self.quantize(result)

    def add(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.add(x, y)
        return self.quantize(result)

    def subtract(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.sub(x, y)
        return self.quantize(result)

    def floor_divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        result = torch.div(x, y, rounding_mode='floor')
        return self.quantize(result)

    def bitwise_and(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_and(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    def bitwise_or(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_or(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    def bitwise_xor(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_xor(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        x = self.quantize(x)
        result = torch.sum(x, dim=axis)
        return self.quantize(result)

    def prod(self, x, axis=None):
        x = self.quantize(x)
        result = torch.prod(x, dim=axis)
        return self.quantize(result)

    def mean(self, x, axis=None):
        x = self.quantize(x)
        result = torch.mean(x, dim=axis)
        return self.quantize(result)

    def std(self, x, axis=None):
        x = self.quantize(x)
        result = torch.std(x, dim=axis)
        return self.quantize(result)

    def var(self, x, axis=None):
        x = self.quantize(x)
        result = torch.var(x, dim=axis)
        return self.quantize(result)

    def dot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.dot(x, y)
        return self.quantize(result)

    def matmul(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.matmul(x, y)
        return self.quantize(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        x = self.quantize(x)
        result = torch.floor(x)
        return self.quantize(result)

    def ceil(self, x):
        x = self.quantize(x)
        result = torch.ceil(x)
        return self.quantize(result)

    def round(self, x, decimals=0):
        x = self.quantize(x)
        if decimals == 0:
            result = torch.round(x)
        else:
            factor = 10 ** decimals
            result = torch.round(x * factor) / factor
        return self.quantize(result)

    def sign(self, x):
        x = self.quantize(x)
        result = torch.sign(x)
        return self.quantize(result)

    def clip(self, x, a_min, a_max):
        a_min = torch.tensor(a_min, dtype=torch.float32, device=x.device)
        a_max = torch.tensor(a_max, dtype=torch.float32, device=x.device)
        x = self.quantize(x)
        chopped_a_min = self.quantize(a_min)
        chopped_a_max = self.quantize(a_max)
        result = torch.clamp(x, min=chopped_a_min, max=chopped_a_max)
        return self.quantize(result)

    # Special Functions
    def erf(self, x):
        x = self.quantize(x)
        result = torch.erf(x)
        return self.quantize(result)

    def erfc(self, x):
        x = self.quantize(x)
        result = torch.erfc(x)
        return self.quantize(result)

    def gamma(self, x):
        x = self.quantize(x)
        result = torch.special.gamma(x)
        return self.quantize(result)

    # New Mathematical Functions
    def fabs(self, x):
        x = self.quantize(x)
        result = torch.abs(x)
        return self.quantize(result)

    def logaddexp(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.logaddexp(x, y)
        return self.quantize(result)

    def cumsum(self, x, axis=None):
        x = self.quantize(x)
        result = torch.cumsum(x, dim=axis)
        return self.quantize(result)

    def cumprod(self, x, axis=None):
        x = self.quantize(x)
        result = torch.cumprod(x, dim=axis)
        return self.quantize(result)

    def degrees(self, x):
        x = self.quantize(x)
        result = torch.deg2rad(x) * (180 / torch.pi)
        return self.quantize(result)

    def radians(self, x):
        x = self.quantize(x)
        result = torch.rad2deg(x) * (torch.pi / 180)
        return self.quantize(result)
    
