import numpy as np
from typing import Tuple


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
                  abs_x: np.ndarray) -> np.ndarray:
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
            
        Returns
        ----------  
        result : numpy.ndarray
            Quantized tensor in fixed-point representation
        """
        


        scaled = abs_x / self.resolution

        if self.rmode in {"nearest", 1}:
            quantized = np.round(scaled)

        elif self.rmode in {"plus_inf", 2}:
            quantized = np.where(sign > 0, np.ceil(scaled), np.floor(scaled))

        elif self.rmode in {"minus_inf", 3}:
            quantized = np.where(sign > 0, np.floor(scaled), np.ceil(scaled))

        elif self.rmode in {"towards_zero", 4}:
            quantized = np.trunc(scaled)

        elif self.rmode in {"stoc_prop", 5}:
            floor_val = np.floor(scaled)
            fraction = scaled - floor_val
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < fraction, floor_val + 1, floor_val)

        elif self.rmode in {"stoc_equal", 6}:
            floor_val = np.floor(scaled)
            prob = np.random.random(scaled.shape)
            quantized = np.where(prob < 0.5, floor_val, floor_val + 1)

        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")

        result = sign * quantized * self.resolution
        result = np.clip(result, self.min_value, self.max_value)

        result[np.isinf(x)] = np.sign(x[np.isinf(x)]) * self.max_value
        result[np.isnan(x)] = np.nan

        return result

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Convert floating-point tensor to fixed-point representation with specified rounding method.
        
        Parameters
        ----------  
        x : numpy.ndarray
            Input tensor
            
        Returns
        ----------  
        result : numpy.ndarray
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
