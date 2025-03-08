import jax.numpy as jnp
import jax.random as random
from typing import Tuple
from jax import jit

class FPRound:
    def __init__(self, ibits: int, fbits: int):
        """Initialize fixed-point simulator with Qm.n format."""
        self.ibits = ibits
        self.fbits = fbits
        self.total_bits = ibits + fbits
        self.max_value = 2 ** (ibits - 1) - 2 ** (-fbits)
        self.min_value = -2 ** (ibits - 1)
        self.resolution = 2 ** (-fbits)

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
                  rmode: str,
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
            
        rmode : str | int
            - 0 or "nearest_odd": Round to nearest value, ties to odd (Not implemented). 
            - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
            - 2 or "plus_inf": Round towards plus infinity (round up).
            - 3 or "minus_inf": Round towards minus infinity (round down).
            - 4 or "toward_zero": Truncate toward zero (no rounding up).
            - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
            - 6 or "stoc_equal": Stochastic rounding with 50% probability.

        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        scaled = abs_x / self.resolution

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

        result = sign * quantized * self.resolution
        result = jnp.clip(result, self.min_value, self.max_value)

        result = jnp.where(jnp.isinf(x), jnp.sign(x) * self.max_value, result)
        result = jnp.where(jnp.isnan(x), jnp.nan, result)

        return result

    @jit
    def quantize(self, x: jnp.ndarray, rmode: str = "nearest", 
                 key: random.PRNGKey = None) -> jnp.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
                        
        rmode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        sign, abs_x = self._to_fixed_point_components(x)
        return self._quantize(x, sign, abs_x, rmode, key)

    @staticmethod
    def _quantize_static(sim: 'FixedPointSimulator', 
                         x: jnp.ndarray,
                         sign: jnp.ndarray,
                         abs_x: jnp.ndarray,
                         rmode: str,
                         key: random.PRNGKey = None) -> jnp.ndarray:
        """Static quantization method for JIT compilation."""
        scaled = abs_x / sim.resolution

        if rmode == "nearest":
            quantized = jnp.round(scaled)
        elif rmode == "up":
            quantized = jnp.where(sign > 0, jnp.ceil(scaled), jnp.floor(scaled))
        elif rmode == "down":
            quantized = jnp.where(sign > 0, jnp.floor(scaled), jnp.ceil(scaled))
        elif rmode == "towards_zero":
            quantized = jnp.trunc(scaled)
        elif rmode == "stochastic_equal":
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < 0.5, floor_val, floor_val + 1)
        elif rmode == "stochastic_proportional":
            if key is None:
                raise ValueError("PRNG key required for stochastic rounding")
            floor_val = jnp.floor(scaled)
            fraction = scaled - floor_val
            prob = random.uniform(key, scaled.shape)
            quantized = jnp.where(prob < fraction, floor_val + 1, floor_val)
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")

        result = sign * quantized * sim.resolution
        result = jnp.clip(result, sim.min_value, sim.max_value)

        result = jnp.where(jnp.isinf(x), jnp.sign(x) * sim.max_value, result)
        result = jnp.where(jnp.isnan(x), jnp.nan, result)

        return result

    def quantize(self, x: jnp.ndarray, rmode: str = "nearest", 
                 key: random.PRNGKey = None) -> jnp.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
                        
        rmode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        
        sign, abs_x = self._to_fixed_point_components(x)
        return self._quantize_static(self, x, sign, abs_x, rmode, key)

    def get_format_info(self) -> dict:
        """Return information about the fixed-point format."""
        return {
            "format": f"Q{self.ibits}.{self.fbits}",
            "total_bits": self.total_bits,
            "range": (self.min_value, self.max_value),
            "resolution": self.resolution
        }


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
