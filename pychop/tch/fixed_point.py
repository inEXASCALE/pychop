import torch
from typing import Tuple

class FPRound:
    def __init__(self, integer_bits: int=4, fractional_bits: int=4):
        """
        Initialize fixed-point simulator.
        
        Parameters
        ----------  
        integer_bits : int, default=4
            The bitwidth of integer part. 
    
        fractional_bit : int, default=4
            The bitwidth of fractional part. 
            
        """
        
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.total_bits = integer_bits + fractional_bits
        self.max_value = 2 ** (integer_bits - 1) - 2 ** (-fractional_bits)
        self.min_value = -2 ** (integer_bits - 1)
        self.resolution = 2 ** (-fractional_bits)

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
                  abs_x: torch.Tensor,
                  rounding_mode: str) -> torch.Tensor:
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
            
        rounding_mode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        
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
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
                        
        rounding_mode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        
        sign, abs_x = self._to_fixed_point_components(x)
        return self._quantize(x, sign, abs_x, rounding_mode)
        
    def get_format_info(self) -> dict:
        """Return information about the fixed-point format."""
        return {
            "format": f"Q{self.integer_bits}.{self.fractional_bits}",
            "total_bits": self.total_bits,
            "range": (self.min_value, self.max_value),
            "resolution": self.resolution
        }



