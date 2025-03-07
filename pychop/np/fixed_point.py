import numpy as np
from typing import Tuple



class FPRound:
    def __init__(self, ibits: int, fbits: int):
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
                  abs_x: np.ndarray,
                  rmode: str) -> np.ndarray:
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
            
        rmode : str | int
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : numpy.ndarray
            Quantized tensor in fixed-point representation
        """
        
        scaled = abs_x / self.resolution

        if rmode in {"nearest", 1}:
            quantized = np.round(scaled)
        elif rmode in {"up", 2}:
            quantized = np.where(sign > 0, np.ceil(scaled), np.floor(scaled))
        elif rmode in {"down", 3}:
            quantized = np.where(sign > 0, np.floor(scaled), np.ceil(scaled))
        elif rmode in {"towards_zero", 4}:
            quantized = np.trunc(scaled)
        elif rmode in {"stochastic_equal", 5}:
            floor_val = np.floor(scaled)
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < 0.5, floor_val, floor_val + 1)
        elif rmode in {"stochastic_proportional", 6}:
            floor_val = np.floor(scaled)
            fraction = scaled - floor_val
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < fraction, floor_val + 1, floor_val)
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")

        result = sign * quantized * self.resolution
        result = np.clip(result, self.min_value, self.max_value)

        result[np.isinf(x)] = np.sign(x[np.isinf(x)]) * self.max_value
        result[np.isnan(x)] = np.nan

        return result

    def quantize(self, x: np.ndarray, rmode: str = "nearest") -> np.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : numpy.ndarray
            Input tensor
                        
        rmode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : numpy.ndarray
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
