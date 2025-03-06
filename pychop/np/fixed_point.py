import numpy as np
from typing import Tuple


def round_clamp(x, bits=8):
    x = np.clip(x, a_min=0, a_max=2**(bits)-1)
    x = np.round(x * 2**(bits)) / (2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    """
    Parameters
    ----------
    x : numpy.ndarray,
        The input array.    
    
    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    Methods
    ----------
    x_q : numpy.ndarray,
        The quantized array.
    """
    x_f = np.sign(x)*round_clamp(np.abs(x) - np.floor(np.abs(x)), fbits)
    x_i = np.sign(x)*round_clamp(np.floor(np.abs(x)), ibits)
    return (x_i + x_f)




class FPRound:
    def __init__(self, integer_bits: int, fractional_bits: int):
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
                  rounding_mode: str) -> np.ndarray:
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
            
        rounding_mode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : numpy.ndarray
            Quantized tensor in fixed-point representation
        """
        
        scaled = abs_x / self.resolution

        if rounding_mode == "nearest":
            quantized = np.round(scaled)
        elif rounding_mode == "up":
            quantized = np.where(sign > 0, np.ceil(scaled), np.floor(scaled))
        elif rounding_mode == "down":
            quantized = np.where(sign > 0, np.floor(scaled), np.ceil(scaled))
        elif rounding_mode == "towards_zero":
            quantized = np.trunc(scaled)
        elif rounding_mode == "stochastic_equal":
            floor_val = np.floor(scaled)
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < 0.5, floor_val, floor_val + 1)
        elif rounding_mode == "stochastic_proportional":
            floor_val = np.floor(scaled)
            fraction = scaled - floor_val
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < fraction, floor_val + 1, floor_val)
        else:
            raise ValueError(f"Unsupported rounding mode: {rounding_mode}")

        result = sign * quantized * self.resolution
        result = np.clip(result, self.min_value, self.max_value)

        result[np.isinf(x)] = np.sign(x[np.isinf(x)]) * self.max_value
        result[np.isnan(x)] = np.nan

        return result

    def quantize(self, x: np.ndarray, rounding_mode: str = "nearest") -> np.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : numpy.ndarray
            Input tensor
                        
        rounding_mode : str
            One of 'nearest', 'up', 'down', 'towards_zero', 
            'stochastic_equal', 'stochastic_proportional'
        
        Returns
        ----------  
        result : numpy.ndarray
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
