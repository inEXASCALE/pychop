import torch
from typing import Tuple

class FPRound:
    def __init__(self, ibits: int=4, fbits: int=4):
        """
        Initialize fixed-point simulator.
        
        Parameters
        ----------  
        ibits : int, default=4
            The bitwidth of integer part. 
    
        fractional_bit : int, default=4
            The bitwidth of fractional part. 
            
        """
        
        self.ibits = ibits
        self.fbits = fbits
        self.total_bits = ibits + fbits
        self.max_value = 2 ** (ibits - 1) - 2 ** (-fbits)
        self.min_value = -2 ** (ibits - 1)
        self.resolution = 2 ** (-fbits)

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
                  rmode: str) -> torch.Tensor:
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
            
        rmode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : torch.Tensor
            Quantized tensor in fixed-point representation
        """
        
        scaled = abs_x / self.resolution

        if rmode in {"nearest", 1}:
            quantized = torch.round(scaled)
        elif rmode in {"up", 2}:
            quantized = torch.where(sign > 0, torch.ceil(scaled), torch.floor(scaled))
        elif rmode in {"down", 3}:
            quantized = torch.where(sign > 0, torch.floor(scaled), torch.ceil(scaled))
        elif rmode in {"towards_zero", 4}:
            quantized = torch.trunc(scaled)
        elif rmode in {"stochastic_equal", 5}:
            floor_val = torch.floor(scaled)
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < 0.5, floor_val, floor_val + 1)
        elif rmode in {"stochastic_proportional", 6}:
            floor_val = torch.floor(scaled)
            fraction = scaled - floor_val
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < fraction, floor_val + 1, floor_val)
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")

        result = sign * quantized * self.resolution
        result = torch.clamp(result, self.min_value, self.max_value)

        result[torch.isinf(x)] = torch.sign(x[torch.isinf(x)]) * self.max_value
        result[torch.isnan(x)] = float('nan')

        return result

    def quantize(self, x: torch.Tensor, rmode: str = "nearest") -> torch.Tensor:
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
        return self._quantize(x, sign, abs_x, rmode)
        
    def get_format_info(self) -> dict:
        """Return information about the fixed-point format."""
        return {
            "format": f"Q{self.ibits}.{self.fbits}",
            "total_bits": self.total_bits,
            "range": (self.min_value, self.max_value),
            "resolution": self.resolution
        }



