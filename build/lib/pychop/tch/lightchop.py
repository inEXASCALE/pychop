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
        - 1 : Round to nearest value, ties to even (IEEE 754 default).
        - 2 : Round towards plus infinity (round up).
        - 3 : Round towards minus infinity (round down).
        - 4 : Truncate toward zero (no rounding up).
        - 5 : Stochastic rounding proportional to the fractional part.
        - 6 : Stochastic rounding with 50% probability.
        - 7 : Round to nearest value, ties to zero.
        - 8 : Round to nearest value, ties to away.
        - 9 : Round to odd.
        
    random_state : int, default=42
        random seed for stochastic rounding.
    """
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True, chunk_size: int = 1000, random_state: int = 42):
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
        self.inv_sig_steps = 1.0 / self.sig_steps
        self.inv_min_exp_power = 1.0 / self.min_exp_power  # Precompute for subnormal case
        torch.manual_seed(random_state)
        
    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to custom float representation with proper IEEE 754 handling."""
        sign = torch.sgn(x) if x.is_complex() else torch.sign(x)
        abs_x = torch.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        exponent = torch.floor(torch.log2(abs_x.clamp(min=1e-38)))
        significand = abs_x * (2.0 ** -exponent)
        
        subnormal_mask = (exponent < self.min_exp)
        if self.subnormal:
            significand = torch.where(subnormal_mask, abs_x * self.inv_min_exp_power, significand)
            exponent = torch.where(subnormal_mask, self.min_exp, exponent)
        else:
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
        """Quantize components according to IEEE 754 rules with specified rounding mode."""
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
        
        elif self.rmode == 9:  # Round-to-Odd
            rounded = torch.round(sig_scaled)
            sig_q = torch.where(rounded % 2 == 0, 
                                rounded + torch.where(sig_scaled >= rounded, 1, -1), 
                                rounded) * inv_sig_steps
            if self.subnormal:
                sub_rounded = torch.round(sub_scaled)
                sig_q = torch.where(subnormal_mask,
                                    torch.where(sub_rounded % 2 == 0,
                                                sub_rounded + torch.where(sub_scaled >= sub_rounded, 1, -1),
                                                sub_rounded) * inv_sig_steps,
                                    sig_q)
        
        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")
        
        subnormal_result = sign * sig_q * self.min_exp_power if self.subnormal else torch.zeros_like(x)
        result = torch.where(normal_mask, sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias)), 
                            torch.where(subnormal_mask, subnormal_result, 
                                    torch.where(inf_mask, sign * float('inf'), 
                                                torch.where(nan_mask, float('nan'), 0.0))))
        
        return result


    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified precision using the initialized rounding mode."""
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask)


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
        self.inv_sig_steps = 1.0 / self.sig_steps
        self.inv_min_exp_power = 1.0 / self.min_exp_power  # Precompute for subnormal case

    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to custom float representation with proper IEEE 754 handling."""
        sign = torch.sgn(x) if x.is_complex() else torch.sign(x)
        abs_x = torch.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        exponent = torch.floor(torch.log2(abs_x.clamp(min=1e-38)))
        significand = abs_x * (2.0 ** -exponent)
        
        subnormal_mask = (exponent < self.min_exp)
        if self.subnormal:
            significand = torch.where(subnormal_mask, abs_x * self.inv_min_exp_power, significand)
            exponent = torch.where(subnormal_mask, self.min_exp, exponent)
        else:
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
        """Quantize components according to IEEE 754 rules with specified rounding mode."""
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
        
        elif self.rmode == 9:  # Round-to-Odd
            rounded = torch.round(sig_scaled)
            sig_q = torch.where(rounded % 2 == 0, 
                                rounded + torch.where(sig_scaled >= rounded, 1, -1), 
                                rounded) * inv_sig_steps
            if self.subnormal:
                sub_rounded = torch.round(sub_scaled)
                sig_q = torch.where(subnormal_mask,
                                    torch.where(sub_rounded % 2 == 0,
                                                sub_rounded + torch.where(sub_scaled >= sub_rounded, 1, -1),
                                                sub_rounded) * inv_sig_steps,
                                    sig_q)
        
        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")
        
        # Fix the condition by handling self.subnormal as a scalar condition
        subnormal_result = sign * sig_q * self.min_exp_power if self.subnormal else torch.zeros_like(x)
        result = torch.where(normal_mask, sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias)), 
                            torch.where(subnormal_mask, subnormal_result, 
                                    torch.where(inf_mask, sign * float('inf'), 
                                                torch.where(nan_mask, float('nan'), 0.0))))
        
        if x.requires_grad:
            result = x + (result - x).detach()
            
        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified precision using the initialized rounding mode."""
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask)

