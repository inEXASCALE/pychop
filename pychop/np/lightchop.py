import numpy as np
from typing import Tuple


class LightChop:
    """
    A class to simulate different floating-point precisions and rounding modes
    for PyTorch tensors. This code implements a custom floating-point precision simulator
    that mimics IEEE 754 floating-point representation with configurable exponent bits (exp_bits),
    significand bits (sig_bits), and various rounding modes (rmode). 
    It uses PyTorch tensors for efficient computation and handles special cases like zeros,
    infinities, NaNs, and subnormal numbers. The code follows IEEE 754 conventions for sign, 
    exponent bias, implicit leading 1 (for normal numbers), and subnormal number handling.

    Initialize with specific format parameters.
    Convert to custom float representation with proper IEEE 754 handling
    
    Parameters
    ----------
    exp_bits: int 
        Number of bits for exponent.

    sig_bits : int
        Number of bits for significand (significant digits)

    rmode : int
        rounding modes.

        Rounding mode to use when quantizing the significand. Options are:
        - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
        - 2 or "plus_inf": Round towards plus infinity (round up).
        - 3 or "minus_inf": Round towards minus infinity (round down).
        - 4 or "toward_zero": Truncate toward zero (no rounding up).
        - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
        - 6 or "stoc_equal": Stochastic rounding with 50% probability.
        - 7 or "nearest_ties_to_zero": Round to nearest value, ties to zero.
        - 8 or "nearest_ties_to_away": Round to nearest value, ties to away.

    random_state : int, default=42
        random seed for stochastic rounding.
    """

    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, 
                 subnormal: bool = True, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.subnormal = subnormal
        
        np.random.seed(random_state)
        
    def _to_custom_float(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray, np.ndarray]:
        sign = np.sign(x)
        abs_x = np.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = np.isinf(x)
        nan_mask = np.isnan(x)
        
        exponent = np.floor(np.log2(np.maximum(abs_x, 2.0**-24)))
        significand = abs_x / (2.0 ** exponent)
        
        if self.subnormal:
            subnormal_mask = (exponent < self.min_exp)
            significand = np.where(subnormal_mask,
                                abs_x / (2.0 ** self.min_exp),
                                significand)
            exponent = np.where(subnormal_mask,
                              self.min_exp,
                              exponent)
        else:
            # Flush subnormals to zero
            subnormal_mask = (exponent < self.min_exp)
            significand = np.where(subnormal_mask, 0.0, significand)
            exponent = np.where(subnormal_mask, 0, exponent)
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask
    
    def _quantize_components(self, 
                           x: np.ndarray,
                           sign: np.ndarray, 
                           exponent: np.ndarray, 
                           significand: np.ndarray,
                           zero_mask: np.ndarray,
                           inf_mask: np.ndarray,
                           nan_mask: np.ndarray,
                           rmode: str) -> np.ndarray:
        
        exp_min = 0
        exp_max = 2**self.exp_bits - 1
        exponent = np.clip(exponent, exp_min, exp_max)
        
        significand_steps = 2 ** self.sig_bits
        normal_mask = (exponent > 0) & (exponent < exp_max)
        subnormal_mask = (exponent == 0) & (significand > 0) if self.subnormal else np.zeros_like(x, dtype=bool)
        significand_normal = significand - 1.0
        
        if rmode in {"nearest", 1}:
            significand_q = np.round(significand_normal * significand_steps) / significand_steps
            significand_q = np.where(subnormal_mask,
                                   np.round(significand * significand_steps) / significand_steps,
                                   significand_q)
                
        elif rmode in {"plus_inf", 2}:
            significand_q = np.where(sign > 0,
                                   np.ceil(significand_normal * significand_steps),
                                   np.floor(significand_normal * significand_steps)) / significand_steps
            significand_q = np.where(subnormal_mask,
                                   np.where(sign > 0,
                                          np.ceil(significand * significand_steps),
                                          np.floor(significand * significand_steps)) / significand_steps,
                                   significand_q)
                
        elif rmode in {"minus_inf", 3}:
            significand_q = np.where(sign > 0,
                                   np.floor(significand_normal * significand_steps),
                                   np.ceil(significand_normal * significand_steps)) / significand_steps
            significand_q = np.where(subnormal_mask,
                                   np.where(sign > 0,
                                          np.floor(significand * significand_steps),
                                          np.ceil(significand * significand_steps)) / significand_steps,
                                   significand_q)
                
        elif rmode in {"towards_zero", 4}:
            significand_q = np.floor(significand_normal * significand_steps) / significand_steps
            significand_q = np.where(subnormal_mask,
                                   np.floor(significand * significand_steps) / significand_steps,
                                   significand_q)
                
        elif rmode in {"stoc_prop", 5}:
            significand_scaled = significand_normal * significand_steps
            floor_val = np.floor(significand_scaled)
            fraction = significand_scaled - floor_val
            prob = np.random.random(significand_scaled.shape)
            significand_q = np.where(prob < fraction, floor_val + 1, floor_val) / significand_steps
            significand_q = np.where(subnormal_mask,
                                   np.where(prob < (significand * significand_steps - np.floor(significand * significand_steps)),
                                          np.floor(significand * significand_steps) + 1,
                                          np.floor(significand * significand_steps)) / significand_steps,
                                   significand_q)
                
        elif rmode in {"stoc_equal", 6}:
            significand_scaled = significand_normal * significand_steps
            floor_val = np.floor(significand_scaled)
            prob = np.random.random(significand_scaled.shape)
            significand_q = np.where(prob < 0.5, floor_val, floor_val + 1) / significand_steps
            significand_q = np.where(subnormal_mask,
                                   np.where(prob < 0.5,
                                          np.floor(significand * significand_steps),
                                          np.floor(significand * significand_steps) + 1) / significand_steps,
                                   significand_q)
                
        elif rmode in {"nearest_ties_to_zero", 7}:
            significand_scaled = significand_normal * significand_steps
            floor_val = np.floor(significand_scaled)
            ceil_val = np.ceil(significand_scaled)
            is_half = np.abs(significand_scaled - floor_val - 0.5) < 1e-6
            significand_q = np.where(
                is_half,
                np.where(sign >= 0, floor_val, ceil_val),
                np.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = np.floor(significand_subnormal)
            sub_ceil = np.ceil(significand_subnormal)
            sub_is_half = np.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = np.where(
                subnormal_mask,
                np.where(
                    sub_is_half,
                    np.where(sign >= 0, sub_floor, sub_ceil),
                    np.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
            
        elif rmode in {"nearest_ties_to_away", 8}:
            significand_scaled = significand_normal * significand_steps
            floor_val = np.floor(significand_scaled)
            ceil_val = np.ceil(significand_scaled)
            is_half = np.abs(significand_scaled - floor_val - 0.5) < 1e-6
            significand_q = np.where(
                is_half,
                np.where(sign >= 0, ceil_val, floor_val),
                np.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = np.floor(significand_subnormal)
            sub_ceil = np.ceil(significand_subnormal)
            sub_is_half = np.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = np.where(
                subnormal_mask,
                np.where(
                    sub_is_half,
                    np.where(sign >= 0, sub_ceil, sub_floor),
                    np.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
    
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")
        
        result = np.zeros_like(x)
        result = np.where(normal_mask,
                        sign * (1.0 + significand_q) * (2.0 ** (exponent - self.bias)),
                        result)
        if self.subnormal:
            result = np.where(subnormal_mask,
                            sign * significand_q * (2.0 ** self.min_exp),
                            result)
        result = np.where(zero_mask, 0.0, result)
        result = np.where(inf_mask, np.sign(x) * np.inf, result)
        result = np.where(nan_mask, np.nan, result)
        
        return result

    def quantize(self, x: np.ndarray) -> np.ndarray:
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode)

    def __call__(self, x: np.ndarray):
        return self.quantize(x)
