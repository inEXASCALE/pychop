import jax.numpy as jnp
import jax.random as random
from typing import Tuple
from jax import jit

def round_clamp(x, bits=8):
    x = jnp.clip(x, a_min=0, a_max=2**(bits)-1)
    x = jnp.round(x * 2**(bits)) / (2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    """
    Parameters
    ----------
    x : numpy.ndarray | jax.Array,
        The input array.    
    
    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    Methods
    ----------
    x_q : numpy.ndarray | jax.Array,
        The quantized array.
    """
    x_f = jnp.sign(x)*round_clamp(jnp.abs(x) - jnp.floor(jnp.abs(x)), fbits)
    x_i = jnp.sign(x)*round_clamp(jnp.floor(jnp.abs(x)), ibits)
    return (x_i + x_f)




class FPRounding:
    def __init__(self, integer_bits: int, fractional_bits: int):
        """
        Initialize fixed-point simulator with Qm.n format.
        
        Args:
            integer_bits: Number of bits for integer part (including sign bit), m in Qm.n
            fractional_bits: Number of bits for fractional part, n in Qm.n
        """
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.total_bits = integer_bits + fractional_bits
        self.max_value = 2 ** (integer_bits - 1) - 2 ** (-fractional_bits)
        self.min_value = -2 ** (integer_bits - 1)
        self.resolution = 2 ** (-fractional_bits)

    def _to_fixed_point_components(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Extract sign and magnitude from floating-point input."""
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        return sign, abs_x

    def _quantize(self, 
                  x: jnp.ndarray,
                  sign: jnp.ndarray,
                  abs_x: jnp.ndarray,
                  rounding_mode: str,
                  key: random.PRNGKey = None) -> jnp.ndarray:
        """Quantize to fixed-point Qm.n with specified rounding mode."""
        scaled = abs_x / self.resolution

        if rounding_mode == "nearest":
            quantized = jnp.round(scaled)
        elif rounding_mode == "up":
            quantized = jnp.where(sign > 0, jnp.ceil(scaled), jnp.floor(scaled))
        elif rounding_mode == "down":
            quantized = jnp.where(sign > 0, jnp.floor(scaled), jnp.ceil(scaled))
        elif rounding_mode == "towards_zero":
            quantized = jnp.trunc(scaled)
        elif rounding_mode == "stochastic_equal":
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < 0.5, floor_val, floor_val + 1)
        elif rounding_mode == "stochastic_proportional":
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            fraction = scaled - floor_val
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < fraction, floor_val + 1, floor_val)
        else:
            raise ValueError(f"Unsupported rounding mode: {rounding_mode}")

        result = sign * quantized * self.resolution
        result = jnp.clip(result, self.min_value, self.max_value)

        result = jnp.where(jnp.isinf(x), jnp.sign(x) * self.max_value, result)
        result = jnp.where(jnp.isnan(x), jnp.nan, result)

        return result

    @jit
    def quantize(self, x: jnp.ndarray, rounding_mode: str = "nearest", 
                 key: random.PRNGKey = None) -> jnp.ndarray:
        """Convert floating-point array to fixed-point Qm.n representation."""
        sign, abs_x = self._to_fixed_point_components(x)
        return self._quantize(x, sign, abs_x, rounding_mode, key)
