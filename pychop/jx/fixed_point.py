import jax.numpy as jnp
import jax.random as random
from typing import Tuple
from jax import jit

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

    def _to_fixed_point_components(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        return sign, abs_x

    def _quantize(self, 
                  x: jnp.ndarray,
                  sign: jnp.ndarray,
                  abs_x: jnp.ndarray,
                  key: random.PRNGKey = None) -> jnp.ndarray:
        """
        Quantize to fixed-point with specified rounding mode.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
            
        sign : torch.Tensor
            Signs of input values
            
        abs_x : torch.Tensor
            Absolute values of input
            
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        scaled = abs_x / self.resolution

        if self.rmode in {"nearest", 1}:
            quantized = jnp.round(scaled)

        elif self.rmode in {"plus_inf", 2}:
            quantized = jnp.where(sign > 0, jnp.ceil(scaled), jnp.floor(scaled))

        elif self.rmode in {"minus_inf", 3}:
            quantized = jnp.where(sign > 0, jnp.floor(scaled), jnp.ceil(scaled))

        elif self.rmode in {"towards_zero", 4}:
            quantized = jnp.trunc(scaled)

        elif self.rmode in {"stoc_prop", 5}:
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            fraction = scaled - floor_val
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < fraction, floor_val + 1, floor_val)

        elif self.rmode in {"stoc_equal", 6}:
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < 0.5, floor_val, floor_val + 1)

        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")

        result = sign * quantized * self.resolution
        result = jnp.clip(result, self.min_value, self.max_value)

        result = jnp.where(jnp.isinf(x), jnp.sign(x) * self.max_value, result)
        result = jnp.where(jnp.isnan(x), jnp.nan, result)

        return result

    @jit
    def quantize(self, x: jnp.ndarray, key: random.PRNGKey = None) -> jnp.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
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
        return self._quantize(x, sign, abs_x, key)

    @staticmethod
    def _quantize_static(sim: 'FixedPointSimulator', 
                         x: jnp.ndarray,
                         sign: jnp.ndarray,
                         abs_x: jnp.ndarray,
                         rmode: int,
                         key: random.PRNGKey = None) -> jnp.ndarray:
        """Static quantization method for JIT compilation."""
        scaled = abs_x / sim.resolution

        if rmode in {"nearest", 1}:
            quantized = jnp.round(scaled)
            
        elif rmode in {"plus_inf", 2}:
            quantized = jnp.where(sign > 0, jnp.ceil(scaled), jnp.floor(scaled))
            
        elif rmode in {"minus_inf", 3}:
            quantized = jnp.where(sign > 0, jnp.floor(scaled), jnp.ceil(scaled))
            
        elif rmode in {"towards_zero", 4}:
            quantized = jnp.trunc(scaled)

        elif rmode in {"stoc_prop", 5}:
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            fraction = scaled - floor_val
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < fraction, floor_val + 1, floor_val)
            
        elif rmode in {"stoc_equal", 6}:
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < 0.5, floor_val, floor_val + 1)
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")

        result = sign * quantized * sim.resolution
        result = jnp.clip(result, sim.min_value, sim.max_value)

        result = jnp.where(jnp.isinf(x), jnp.sign(x) * sim.max_value, result)
        result = jnp.where(jnp.isnan(x), jnp.nan, result)

        return result

    def quantize(self, x: jnp.ndarray, key: random.PRNGKey = None) -> jnp.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
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
        return self._quantize_static(self, x, sign, abs_x, self.rmode, key)

    
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
    
# Test the implementation
def test_fixed_point():
    values = jnp.array([1.7641, 0.3097, -0.2021, 2.4700, 0.3300])
    fx_sim = FixedPointSimulator(4, 4)
    # Print format info
    info = fx_sim.get_format_info()
    print(f"Format: {info['format']}")
    print(f"Range: {info['range']}")
    print(f"Resolution: {info['resolution']}")
    print()
    
    print("Input values:", values)
    rmodes = ["nearest", "up", "down", "towards_zero", 
                      "stochastic_equal", "stochastic_proportional"]
    key = random.PRNGKey(42)
    for mode in rmodes:
        if "stochastic" in mode:
            result = fx_sim.quantize(values, mode, key)
        else:
            result = fx_sim.quantize(values, mode)
        print(f"{mode:20}: {result}")

if __name__ == "__main__":
    test_fixed_point()
