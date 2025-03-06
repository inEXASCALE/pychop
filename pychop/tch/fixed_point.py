import torch

from typing import Tuple

def round_clamp(x, bits=8):
    x = x.clamp(0,2**(bits)-1)
    x = x.mul(2**(bits)).round().div(2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    """
    Parameters
    ----------
    x : torch.Tensor,
        The input array.    
    
    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    Methods
    ----------
    x_q : torch.Tensor,
        The quantized array.
    """

    x_f = x.sign()*round_clamp(torch.abs(x) - torch.abs(x).floor(), fbits)
    x_i = x.sign()*round_clamp(x.abs().floor(), ibits)
    return (x_i + x_f)




class FPRounding:
    def __init__(self, integer_bits: int, fractional_bits: int):
        """Initialize fixed-point simulator with Qm.n format."""
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.total_bits = integer_bits + fractional_bits
        self.max_value = 2 ** (integer_bits - 1) - 2 ** (-fractional_bits)
        self.min_value = -2 ** (integer_bits - 1)
        self.resolution = 2 ** (-fractional_bits)

    def _to_fixed_point_components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract sign and magnitude from floating-point input."""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        return sign, abs_x

    def _quantize(self, 
                  x: torch.Tensor,
                  sign: torch.Tensor,
                  abs_x: torch.Tensor,
                  rounding_mode: str) -> torch.Tensor:
        """Quantize to fixed-point Qm.n with specified rounding mode."""
        scaled = abs_x / self.resolution

        if rounding_mode == "nearest":
            quantized = torch.round(scaled)
        elif rounding_mode == "up":
            quantized = torch.where(sign > 0, torch.ceil(scaled), torch.floor(scaled))
        elif rounding_mode == "down":
            quantized = torch.where(sign > 0, torch.floor(scaled), torch.ceil(scaled))
        elif rounding_mode == "towards_zero":
            quantized = torch.trunc(scaled)
        elif rounding_mode == "stochastic_equal":
            floor_val = torch.floor(scaled)
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < 0.5, floor_val, floor_val + 1)
        elif rounding_mode == "stochastic_proportional":
            floor_val = torch.floor(scaled)
            fraction = scaled - floor_val
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < fraction, floor_val + 1, floor_val)
        else:
            raise ValueError(f"Unsupported rounding mode: {rounding_mode}")

        result = sign * quantized * self.resolution
        result = torch.clamp(result, self.min_value, self.max_value)

        result[torch.isinf(x)] = torch.sign(x[torch.isinf(x)]) * self.max_value
        result[torch.isnan(x)] = float('nan')

        return result

    def quantize(self, x: torch.Tensor, rounding_mode: str = "nearest") -> torch.Tensor:
        """Convert floating-point tensor to fixed-point Qm.n representation."""
        sign, abs_x = self._to_fixed_point_components(x)
        return self._quantize(x, sign, abs_x, rounding_mode)

