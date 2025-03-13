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
    
        fbits : int, default=4
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
        Quantize to fixed-point with specified rounding mode and STE.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor (unquantized, for STE)
            
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
            Quantized tensor in fixed-point representation with STE applied
        """
        scaled = abs_x / self.resolution

        if rmode in {"nearest", 1}:
            quantized = torch.round(scaled)
        elif rmode in {"plus_inf", 2}:
            quantized = torch.where(sign > 0, torch.ceil(scaled), torch.floor(scaled))
        elif rmode in {"minus_inf", 3}:
            quantized = torch.where(sign > 0, torch.floor(scaled), torch.ceil(scaled))
        elif rmode in {"towards_zero", 4}:
            quantized = torch.trunc(scaled)
        elif rmode in {"stoc_prop", 5}:
            floor_val = torch.floor(scaled)
            fraction = scaled - floor_val
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < fraction, floor_val + 1, floor_val)
        elif rmode in {"stoc_equal", 6}:
            floor_val = torch.floor(scaled)
            prob = torch.rand_like(scaled)
            quantized = torch.where(prob < 0.5, floor_val, floor_val + 1)
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")

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

    def quantize(self, x: torch.Tensor, rmode: str = "nearest") -> torch.Tensor:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method and STE.
        
        Parameters
        ----------  
        x : torch.Tensor
            Input tensor
                        
        rmode : str
            One of 'nearest', 'plus_inf', 'minus_inf', 'towards_zero', 
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
