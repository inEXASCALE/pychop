import torch
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

    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1  # Bias for IEEE 754
        self.rmode = rmode
        torch.manual_seed(random_state)

    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
      
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        
        # Handle special cases
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        # Calculate raw exponent and significand
        exponent = torch.floor(torch.log2(abs_x.clamp(min=2.0**-24)))  # Minimum denormal
        
        # Normalize significand to [1, 2)
        significand = abs_x / (2.0 ** exponent)
        
        # Handle subnormals
        subnormal_mask = (exponent < self.min_exp)
        if torch.any(subnormal_mask):
            significand[subnormal_mask] = abs_x[subnormal_mask] / (2.0 ** self.min_exp)
            exponent[subnormal_mask] = self.min_exp
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask
    
    def _quantize_components(self, 
                           x: torch.Tensor,
                           sign: torch.Tensor, 
                           exponent: torch.Tensor, 
                           significand: torch.Tensor,
                           zero_mask: torch.Tensor,
                           inf_mask: torch.Tensor,
                           nan_mask: torch.Tensor,
                           rmode: str) -> torch.Tensor:
        """Quantize components according to IEEE 754 FP16 rules with various rounding modes

        """
        
        # Clamp exponent to representable range (including bias)
        exp_min = 0  # 0 represents subnormals
        exp_max = 2**self.exp_bits - 1  # 31 for FP16
        exponent = exponent.clamp(min=exp_min, max=exp_max)
        
        # Quantize significand
        significand_steps = 2 ** self.sig_bits
        normal_mask = (exponent > 0) & (exponent < exp_max)
        subnormal_mask = (exponent == 0)
        significand_normal = significand - 1.0  # Remove implicit leading 1 for normal numbers
        
        # Apply rounding mode
        if rmode in {"nearest", 1}:
            significand_q = torch.round(significand_normal * significand_steps) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.round(significand[subnormal_mask] * 
                                                       significand_steps) / significand_steps
                
        elif rmode in {"plus_inf", 2}:
            significand_q = torch.where(sign > 0, 
                                   torch.ceil(significand_normal * significand_steps),
                                   torch.floor(significand_normal * significand_steps)) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.where(sign[subnormal_mask] > 0,
                                                       torch.ceil(significand[subnormal_mask] * significand_steps),
                                                       torch.floor(significand[subnormal_mask] * significand_steps)) / significand_steps
                
        elif rmode in {"minus_inf", 3}:
            significand_q = torch.where(sign > 0,
                                   torch.floor(significand_normal * significand_steps),
                                   torch.ceil(significand_normal * significand_steps)) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.where(sign[subnormal_mask] > 0,
                                                       torch.floor(significand[subnormal_mask] * significand_steps),
                                                       torch.ceil(significand[subnormal_mask] * significand_steps)) / significand_steps
                
        elif rmode in {"towards_zero", 4}:
            significand_q = torch.floor(significand_normal * significand_steps) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.floor(significand[subnormal_mask] * significand_steps) / significand_steps
                
        elif rmode in {"stoc_prop", 5}:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            fraction = significand_scaled - floor_val
            prob = torch.rand_like(significand_scaled)
            significand_q = torch.where(prob < fraction, floor_val + 1, floor_val) / significand_steps
            if torch.any(subnormal_mask):
                significand_scaled = significand[subnormal_mask] * significand_steps
                floor_val = torch.floor(significand_scaled)
                fraction = significand_scaled - floor_val
                prob = torch.rand_like(significand_scaled)
                significand_q[subnormal_mask] = torch.where(prob < fraction, floor_val + 1, 
                                                       floor_val) / significand_steps
                
        elif rmode in {"stoc_equal", 6}:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            prob = torch.rand_like(significand_scaled)
            significand_q = torch.where(prob < 0.5, floor_val, floor_val + 1) / significand_steps
            if torch.any(subnormal_mask):
                significand_scaled = significand[subnormal_mask] * significand_steps
                floor_val = torch.floor(significand_scaled)
                prob = torch.rand_like(significand_scaled)
                significand_q[subnormal_mask] = torch.where(prob < 0.5, floor_val, 
                                                       floor_val + 1) / significand_steps
                
        elif self.rmode in {"nearest_ties_to_zero", 7}:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            ceil_val = torch.ceil(significand_scaled)
            is_half = torch.abs(significand_scaled - floor_val - 0.5) < 1e-6  # Robust tie check
            significand_q = torch.where(
                is_half,
                torch.where(sign >= 0, floor_val, ceil_val),  # Toward zero: positive floor, negative ceil
                torch.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = torch.floor(significand_subnormal)
            sub_ceil = torch.ceil(significand_subnormal)
            sub_is_half = torch.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = torch.where(
                subnormal_mask,
                torch.where(
                    sub_is_half,
                    torch.where(sign >= 0, sub_floor, sub_ceil),
                    torch.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
            
        elif self.rmode in {"nearest_ties_to_away", 8}:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            ceil_val = torch.ceil(significand_scaled)
            is_half = torch.abs(significand_scaled - floor_val - 0.5) < 1e-6  # Robust tie check
            significand_q = torch.where(
                is_half,
                torch.where(sign >= 0, ceil_val, floor_val),  # Away from zero: positive ceil, negative floor
                torch.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = torch.floor(significand_subnormal)
            sub_ceil = torch.ceil(significand_subnormal)
            sub_is_half = torch.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = torch.where(
                subnormal_mask,
                torch.where(
                    sub_is_half,
                    torch.where(sign >= 0, sub_ceil, sub_floor),
                    torch.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
    
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")
        
        # Reconstruct the number
        result = torch.zeros_like(x)
        
        # Normal numbers
        if torch.any(normal_mask):
            result[normal_mask] = sign[normal_mask] * (1.0 + significand_q[normal_mask]) * \
                                (2.0 ** (exponent[normal_mask] - self.bias))
        
        # Subnormal numbers
        if torch.any(subnormal_mask):
            result[subnormal_mask] = sign[subnormal_mask] * significand_q[subnormal_mask] * \
                                   (2.0 ** self.min_exp)
        
        # Special cases
        result[zero_mask] = 0.0
        result[inf_mask] = torch.sign(x[inf_mask]) * float('inf')
        result[nan_mask] = float('nan')
        
        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize a tensor to the specified precision with given rounding mode.
        
        Parameters
        ----------
        x: Input tensor
        
        """
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode)


    def __call__(self, x: torch.Tensor):
        return self.quantize(x)







class LightChopSTE:
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True):
            """Initialize float precision simulator with custom format, rounding mode, and subnormal support."""
            self.exp_bits = exp_bits
            self.sig_bits = sig_bits
            self.rmode = rmode
            self.subnormal = subnormal
            self.max_exp = 2 ** (exp_bits - 1) - 1
            self.min_exp = -self.max_exp + 1
            self.bias = 2 ** (exp_bits - 1) - 1
            # Precompute constants
            self.sig_steps = 2 ** sig_bits
            self.min_exp_power = 2.0 ** self.min_exp
            self.exp_min = 0
            self.exp_max = 2 ** exp_bits - 1
            self.inv_sig_steps = 1.0 / self.sig_steps  # Precompute inverse for multiplication

    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to custom float representation with proper IEEE 754 handling."""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        exponent = torch.floor(torch.log2(abs_x.clamp(min=1e-38)))
        significand = abs_x * (2.0 ** -exponent)
        
        if self.subnormal:
            subnormal_mask = (exponent < self.min_exp)
            significand = torch.where(subnormal_mask, abs_x * (self.min_exp_power ** -1), significand)
            exponent = torch.where(subnormal_mask, self.min_exp, exponent)
        else:
            subnormal_mask = (exponent < self.min_exp)
            significand = torch.where(subnormal_mask, 0.0, significand)
            exponent = torch.where(subnormal_mask, 0, exponent)
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask
    
    def _quantize_components(self, 
                           x: torch.Tensor,
                           sign: torch.Tensor, 
                           exponent: torch.Tensor, 
                           significand: torch.Tensor,
                           zero_mask: torch.Tensor,
                           inf_mask: torch.Tensor,
                           nan_mask: torch.Tensor) -> torch.Tensor:
        """Quantize components according to IEEE 754 FP16 rules with specified rounding mode."""
        exponent = torch.clamp(exponent, self.exp_min, self.exp_max)
        
        sig_steps = self.sig_steps
        inv_sig_steps = self.inv_sig_steps
        normal_mask = (exponent > self.exp_min) & (exponent < self.exp_max)
        subnormal_mask = (exponent == self.exp_min) & (significand > 0) if self.subnormal else torch.zeros_like(x, dtype=bool)
        sig_normal = significand - 1.0
        
        sig_scaled = sig_normal * sig_steps
        sub_scaled = significand * sig_steps if self.subnormal else None
        
        if self.rmode == 1:  # Nearest
            sig_q = torch.round(sig_scaled) * inv_sig_steps
            if self.subnormal:
                sig_q = torch.where(subnormal_mask, torch.round(sub_scaled) * inv_sig_steps, sig_q)
            
        elif self.rmode == 2:  # Plus infinity
            sig_q = torch.where(sign > 0, torch.ceil(sig_scaled), torch.floor(sig_scaled)) * inv_sig_steps
            if self.subnormal:
                sig_q = torch.where(subnormal_mask, 
                                   torch.where(sign > 0, torch.ceil(sub_scaled), torch.floor(sub_scaled)) * inv_sig_steps, sig_q)
            
        elif self.rmode == 3:  # Minus infinity
            sig_q = torch.where(sign > 0, torch.floor(sig_scaled), torch.ceil(sig_scaled)) * inv_sig_steps
            if self.subnormal:
                sig_q = torch.where(subnormal_mask, 
                                   torch.where(sign > 0, torch.floor(sub_scaled), torch.ceil(sub_scaled)) * inv_sig_steps, sig_q)
            
        elif self.rmode == 4:  # Towards zero
            sig_q = torch.floor(sig_scaled) * inv_sig_steps
            if self.subnormal:
                sig_q = torch.where(subnormal_mask, torch.floor(sub_scaled) * inv_sig_steps, sig_q)
            
        elif self.rmode == 5:  # Stochastic proportional
            floor_val = torch.floor(sig_scaled)
            fraction = sig_scaled - floor_val
            prob = torch.rand_like(fraction)
            sig_q = torch.where(prob < fraction, floor_val + 1, floor_val) * inv_sig_steps
            if self.subnormal:
                sub_floor = torch.floor(sub_scaled)
                sub_fraction = sub_scaled - sub_floor
                sig_q = torch.where(subnormal_mask, 
                                   torch.where(prob < sub_fraction, sub_floor + 1, sub_floor) * inv_sig_steps, sig_q)
            
        elif self.rmode == 6:  # Stochastic equal
            floor_val = torch.floor(sig_scaled)
            prob = torch.rand_like(floor_val)
            sig_q = torch.where(prob < 0.5, floor_val, floor_val + 1) * inv_sig_steps
            if self.subnormal:
                sub_floor = torch.floor(sub_scaled)
                sig_q = torch.where(subnormal_mask, 
                                   torch.where(prob < 0.5, sub_floor, sub_floor + 1) * inv_sig_steps, sig_q)
            
        elif self.rmode == 7:  # Nearest, ties to zero
            floor_val = torch.floor(sig_scaled)
            is_half = torch.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = torch.where(is_half, torch.where(sign >= 0, floor_val, floor_val + 1), 
                               torch.round(sig_scaled)) * inv_sig_steps
            if self.subnormal:
                sub_floor = torch.floor(sub_scaled)
                sub_is_half = torch.abs(sub_scaled - sub_floor - 0.5) < 1e-6
                sig_q = torch.where(subnormal_mask, 
                                   torch.where(sub_is_half, torch.where(sign >= 0, sub_floor, sub_floor + 1),
                                             torch.round(sub_scaled)) * inv_sig_steps, sig_q)
            
        elif self.rmode == 8:  # Nearest, ties away
            floor_val = torch.floor(sig_scaled)
            is_half = torch.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = torch.where(is_half, torch.where(sign >= 0, floor_val + 1, floor_val), 
                               torch.round(sig_scaled)) * inv_sig_steps
            if self.subnormal:
                sub_floor = torch.floor(sub_scaled)
                sub_is_half = torch.abs(sub_scaled - sub_floor - 0.5) < 1e-6
                sig_q = torch.where(subnormal_mask, 
                                   torch.where(sub_is_half, torch.where(sign >= 0, sub_floor + 1, sub_floor),
                                             torch.round(sub_scaled)) * inv_sig_steps, sig_q)
        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")
        
        result = torch.where(normal_mask, sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias)), 
                           torch.where(subnormal_mask & self.subnormal, sign * sig_q * self.min_exp_power, 
                                     torch.where(inf_mask, torch.sign(x) * float('inf'), 
                                               torch.where(nan_mask, float('nan'), 0.0))))
        
        if x.requires_grad:
            result = x + (result - x).detach()
            
        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified precision using the initialized rounding mode."""
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask)


