import torch
from typing import Tuple
import torch.nn as nn

class LightChop_:
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



    # Trigonometric Functions
    def sin(self, x):
        x = self.quantize(x)
        result = torch.sin(x)
        return self.quantize(result)

    def cos(self, x):
        x = self.quantize(x)
        result = torch.cos(x)
        return self.quantize(result)

    def tan(self, x):
        x = self.quantize(x)
        result = torch.tan(x)
        return self.quantize(result)

    def arcsin(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        result = torch.asin(x)
        return self.quantize(result)

    def arccos(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        result = torch.acos(x)
        return self.quantize(result)

    def arctan(self, x):
        x = self.quantize(x)
        result = torch.atan(x)
        return self.quantize(result)

    # Hyperbolic Functions
    def sinh(self, x):
        x = self.quantize(x)
        result = torch.sinh(x)
        return self.quantize(result)

    def cosh(self, x):
        x = self.quantize(x)
        result = torch.cosh(x)
        return self.quantize(result)

    def tanh(self, x):
        x = self.quantize(x)
        result = torch.tanh(x)
        return self.quantize(result)

    def arcsinh(self, x):
        x = self.quantize(x)
        result = torch.asinh(x)
        return self.quantize(result)

    def arccosh(self, x):
        x = self.quantize(x)
        if not torch.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        result = torch.acosh(x)
        return self.quantize(result)

    def arctanh(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        result = torch.atanh(x)
        return self.quantize(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        x = self.quantize(x)
        result = torch.exp(x)
        return self.quantize(result)

    def expm1(self, x):
        x = self.quantize(x)
        result = torch.expm1(x)
        return self.quantize(result)

    def log(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log input must be positive")
        result = torch.log(x)
        return self.quantize(result)

    def log10(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log10 input must be positive")
        result = torch.log10(x)
        return self.quantize(result)

    def log2(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log2 input must be positive")
        result = torch.log2(x)
        return self.quantize(result)

    def log1p(self, x):
        x = self.quantize(x)
        if not torch.all(x > -1):
            raise ValueError("log1p input must be > -1")
        result = torch.log1p(x)
        return self.quantize(result)

    # Power and Root Functions
    def sqrt(self, x):
        x = self.quantize(x)
        if not torch.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        result = torch.sqrt(x)
        return self.quantize(result)

    def cbrt(self, x):
        x = self.quantize(x)
        result = torch.pow(x, 1/3)
        return self.quantize(result)

    # Miscellaneous Functions
    def abs(self, x):
        x = self.quantize(x)
        result = torch.abs(x)
        return self.quantize(result)

    def reciprocal(self, x):
        x = self.quantize(x)
        if not torch.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        result = torch.reciprocal(x)
        return self.quantize(result)

    def square(self, x):
        x = self.quantize(x)
        result = torch.square(x)
        return self.quantize(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        x = self.quantize(x)
        mantissa, exponent = torch.frexp(x)
        return self.quantize(mantissa), exponent  # Exponent typically not chopped

    def hypot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.hypot(x, y)
        return self.quantize(result)

    def diff(self, x, n=1):
        x = self.quantize(x)
        for _ in range(n):
            x = torch.diff(x)
        return self.quantize(x)

    def power(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.pow(x, y)
        return self.quantize(result)

    def modf(self, x):
        x = self.quantize(x)
        fractional, integer = torch.modf(x)
        return self.quantize(fractional), self.quantize(integer)

    def ldexp(self, x, i):
        x = self.quantize(x)
        i = torch.tensor(i, dtype=torch.int32, device=x.device)  # Exponent not chopped
        result = x * torch.pow(2.0, i)
        return self.quantize(result)

    def angle(self, x):
        if torch.is_complex(x):
            x = self.quantize(x)
            result = torch.angle(x)
        else:
            x = self.quantize(x)
            result = torch.atan2(x, torch.zeros_like(x))
        return self.quantize(result)

    def real(self, x):
        x = self.quantize(x)
        result = torch.real(x) if torch.is_complex(x) else x
        return self.quantize(result)

    def imag(self, x):
        x = self.quantize(x)
        result = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(x)
        return self.quantize(result)

    def conj(self, x):
        x = self.quantize(x)
        result = torch.conj(x) if torch.is_complex(x) else x
        return self.quantize(result)

    def maximum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.maximum(x, y)
        return self.quantize(result)

    def minimum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.minimum(x, y)
        return self.quantize(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.mul(x, y)
        return self.quantize(result)

    def mod(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        result = torch.fmod(x, y)
        return self.quantize(result)

    def divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        result = torch.div(x, y)
        return self.quantize(result)

    def add(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.add(x, y)
        return self.quantize(result)

    def subtract(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.sub(x, y)
        return self.quantize(result)

    def floor_divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        result = torch.div(x, y, rounding_mode='floor')
        return self.quantize(result)

    def bitwise_and(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_and(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    def bitwise_or(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_or(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    def bitwise_xor(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_xor(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        x = self.quantize(x)
        result = torch.sum(x, dim=axis)
        return self.quantize(result)

    def prod(self, x, axis=None):
        x = self.quantize(x)
        result = torch.prod(x, dim=axis)
        return self.quantize(result)

    def mean(self, x, axis=None):
        x = self.quantize(x)
        result = torch.mean(x, dim=axis)
        return self.quantize(result)

    def std(self, x, axis=None):
        x = self.quantize(x)
        result = torch.std(x, dim=axis)
        return self.quantize(result)

    def var(self, x, axis=None):
        x = self.quantize(x)
        result = torch.var(x, dim=axis)
        return self.quantize(result)

    def dot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.dot(x, y)
        return self.quantize(result)

    def matmul(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.matmul(x, y)
        return self.quantize(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        x = self.quantize(x)
        result = torch.floor(x)
        return self.quantize(result)

    def ceil(self, x):
        x = self.quantize(x)
        result = torch.ceil(x)
        return self.quantize(result)

    def round(self, x, decimals=0):
        x = self.quantize(x)
        if decimals == 0:
            result = torch.round(x)
        else:
            factor = 10 ** decimals
            result = torch.round(x * factor) / factor
        return self.quantize(result)

    def sign(self, x):
        x = self.quantize(x)
        result = torch.sign(x)
        return self.quantize(result)

    def clip(self, x, a_min, a_max):
        a_min = torch.tensor(a_min, dtype=torch.float32, device=x.device)
        a_max = torch.tensor(a_max, dtype=torch.float32, device=x.device)
        x = self.quantize(x)
        chopped_a_min = self.quantize(a_min)
        chopped_a_max = self.quantize(a_max)
        result = torch.clamp(x, min=chopped_a_min, max=chopped_a_max)
        return self.quantize(result)

    # Special Functions
    def erf(self, x):
        x = self.quantize(x)
        result = torch.erf(x)
        return self.quantize(result)

    def erfc(self, x):
        x = self.quantize(x)
        result = torch.erfc(x)
        return self.quantize(result)

    def gamma(self, x):
        x = self.quantize(x)
        result = torch.special.gamma(x)
        return self.quantize(result)

    # New Mathematical Functions
    def fabs(self, x):
        x = self.quantize(x)
        result = torch.abs(x)
        return self.quantize(result)

    def logaddexp(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.logaddexp(x, y)
        return self.quantize(result)

    def cumsum(self, x, axis=None):
        x = self.quantize(x)
        result = torch.cumsum(x, dim=axis)
        return self.quantize(result)

    def cumprod(self, x, axis=None):
        x = self.quantize(x)
        result = torch.cumprod(x, dim=axis)
        return self.quantize(result)

    def degrees(self, x):
        x = self.quantize(x)
        result = torch.deg2rad(x) * (180 / torch.pi)
        return self.quantize(result)

    def radians(self, x):
        x = self.quantize(x)
        result = torch.rad2deg(x) * (torch.pi / 180)
        return self.quantize(result)
        
    def __call__(self, x: torch.Tensor):
        return self.quantize(x)



class LightChopSTE(nn.Module):
    """
    A PyTorch module for simulating low-precision floating-point quantization with Quantization-Aware Training (QAT) support.
    
    This implements a FakeQuant-style operator that mimics custom floating-point formats (e.g., FP8 E4M3/E5M2)
    by decomposing into sign/exponent/significand, applying configurable rounding, and supporting subnormals.
    It uses the Straight-Through Estimator (STE) for gradients during training.
    
    Parameters
    ----------
    exp_bits (int): Number of exponent bits (e.g., 4 for E4M3, 5 for E5M2).
    sig_bits (int): Number of significand (mantissa) bits, excluding the implicit leading 1.
    rmode (int): Rounding mode:
        1: Nearest (round ties to even is approximated)
        2: Positive infinity (ceil for positive, floor for negative)
        3: Negative infinity (floor for positive, ceil for negative)
        4: Towards zero (truncation)
        5: Stochastic proportional rounding
        6: Stochastic equal-probability rounding
        7-9: Additional specialized modes (ties handling, round-to-odd, etc.)
    subnormal (bool): Whether to support subnormal (denormal) numbers.
    """
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True):
        super().__init__()
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
        self.inv_min_exp_power = 1.0 / self.min_exp_power

    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        # Clamp to avoid log2(0) or extremely small values
        clamped_abs = abs_x.clamp(min=1e-38)
        exponent = torch.floor(torch.log2(clamped_abs))
        significand = clamped_abs * (2.0 ** -exponent)
        
        subnormal_mask = (exponent < self.min_exp)
        subnormal_allowed = torch.tensor(self.subnormal, device=x.device, dtype=torch.bool)
        
        significand = torch.where(subnormal_mask & subnormal_allowed,
                                  abs_x * self.inv_min_exp_power, significand)
        exponent = torch.where(subnormal_mask & subnormal_allowed,
                               torch.full_like(exponent, float(self.min_exp)), exponent)
        exponent = torch.where(subnormal_mask & ~subnormal_allowed,
                               torch.zeros_like(exponent), exponent)
        significand = torch.where(subnormal_mask & ~subnormal_allowed,
                                  torch.zeros_like(significand), significand)
        
        biased_exponent = exponent + self.bias
        return sign, biased_exponent, significand, zero_mask, inf_mask, nan_mask

    def _quantize_components(self,
                             x: torch.Tensor,
                             sign: torch.Tensor,
                             exponent: torch.Tensor,
                             significand: torch.Tensor,
                             zero_mask: torch.Tensor,
                             inf_mask: torch.Tensor,
                             nan_mask: torch.Tensor) -> torch.Tensor:

        overflow_mask = (exponent > self.exp_max)
        
        exponent = torch.clamp(exponent, self.exp_min, self.exp_max)
        
        sig_steps = self.sig_steps
        inv_sig_steps = self.inv_sig_steps
        
        normal_mask = (exponent > self.exp_min) & (exponent < self.exp_max)
        subnormal_allowed = torch.tensor(self.subnormal, device=x.device, dtype=torch.bool)
        subnormal_mask = (exponent == self.exp_min) & (significand > 0) & subnormal_allowed
        
        sig_normal = significand - 1.0
        sig_scaled = sig_normal * sig_steps
        sub_scaled = significand * sig_steps  # Always compute for safety
        
        # Rounding modes (stochastic only effect when training)
        if self.rmode == 1:  # Nearest
            sig_q = torch.round(sig_scaled)
            sig_q = torch.where(subnormal_mask, torch.round(sub_scaled), sig_q)
        
        elif self.rmode == 2:  # Plus infinity
            sig_q = torch.where(sign > 0, torch.ceil(sig_scaled), torch.floor(sig_scaled))
            sig_q = torch.where(subnormal_mask,
                                torch.where(sign > 0, torch.ceil(sub_scaled), torch.floor(sub_scaled)), sig_q)
        
        elif self.rmode == 3:  # Minus infinity
            sig_q = torch.where(sign > 0, torch.floor(sig_scaled), torch.ceil(sig_scaled))
            sig_q = torch.where(subnormal_mask,
                                torch.where(sign > 0, torch.floor(sub_scaled), torch.ceil(sub_scaled)), sig_q)
        
        elif self.rmode == 4:  # Towards zero
            sig_q = torch.floor(sig_scaled)
            sig_q = torch.where(subnormal_mask, torch.floor(sub_scaled), sig_q)
        
        elif self.rmode == 5:  # Stochastic proportional（training - use random, eval 0 use nearest）
            floor_val = torch.floor(sig_scaled)
            fraction = sig_scaled - floor_val
            if self.training:
                prob = torch.rand_like(fraction)
                sig_q = torch.where(prob < fraction, floor_val + 1, floor_val)
            else:
                sig_q = torch.round(sig_scaled)
            # subnormal
            sub_floor = torch.floor(sub_scaled)
            sub_fraction = sub_scaled - sub_floor
            if self.training:
                prob_sub = torch.rand_like(sub_fraction)
                sig_q = torch.where(subnormal_mask,
                                    torch.where(prob_sub < sub_fraction, sub_floor + 1, sub_floor), sig_q)
            else:
                sig_q = torch.where(subnormal_mask, torch.round(sub_scaled), sig_q)
        
        elif self.rmode == 6:  # Stochastic equal
            floor_val = torch.floor(sig_scaled)
            if self.training:
                prob = torch.rand_like(floor_val)
                sig_q = torch.where(prob < 0.5, floor_val + 1, floor_val)
            else:
                sig_q = torch.round(sig_scaled)
            # subnormal
            sub_floor = torch.floor(sub_scaled)
            if self.training:
                prob_sub = torch.rand_like(sub_floor)
                sig_q = torch.where(subnormal_mask,
                                    torch.where(prob_sub < 0.5, sub_floor + 1, sub_floor), sig_q)
            else:
                sig_q = torch.where(subnormal_mask, torch.round(sub_scaled), sig_q)
        
        elif self.rmode == 7:  # Nearest, ties to zero
            floor_val = torch.floor(sig_scaled)
            is_half = torch.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = torch.where(is_half, torch.where(sign >= 0, floor_val, floor_val + 1),
                                torch.round(sig_scaled))
            sub_floor = torch.floor(sub_scaled)
            sub_is_half = torch.abs(sub_scaled - sub_floor - 0.5) < 1e-6
            sig_q = torch.where(subnormal_mask,
                                torch.where(sub_is_half, torch.where(sign >= 0, sub_floor, sub_floor + 1),
                                            torch.round(sub_scaled)), sig_q)
        
        elif self.rmode == 8:  # Nearest, ties away
            floor_val = torch.floor(sig_scaled)
            is_half = torch.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = torch.where(is_half, torch.where(sign >= 0, floor_val + 1, floor_val),
                                torch.round(sig_scaled))
            sub_floor = torch.floor(sub_scaled)
            sub_is_half = torch.abs(sub_scaled - sub_floor - 0.5) < 1e-6
            sig_q = torch.where(subnormal_mask,
                                torch.where(sub_is_half, torch.where(sign >= 0, sub_floor + 1, sub_floor),
                                            torch.round(sub_scaled)), sig_q)
        
        elif self.rmode == 9:  # Round-to-Odd
            rounded = torch.round(sig_scaled)
            sig_q = torch.where(rounded % 2 == 0,
                                rounded + torch.where(sig_scaled >= rounded, 1, -1), rounded)
            sub_rounded = torch.round(sub_scaled)
            sig_q = torch.where(subnormal_mask,
                                torch.where(sub_rounded % 2 == 0,
                                            sub_rounded + torch.where(sub_scaled >= sub_rounded, 1, -1),
                                            sub_rounded), sig_q)
        
        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")
        
        sig_q = sig_q * inv_sig_steps
        
        # Reconstruct
        normal_result = sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias))
        subnormal_result = sign * sig_q * self.min_exp_power
        
        result = torch.where(normal_mask, normal_result,
                             torch.where(subnormal_mask, subnormal_result,
                                         torch.zeros_like(x)))
        
        # Preserve original special values
        result = torch.where(inf_mask, x, result)
        result = torch.where(nan_mask, x, result)
        result = torch.where(zero_mask, x, result)
        
        # === overflow map to ±Inf ===
        inf_val = sign * torch.full_like(result, float('inf'))
        result = torch.where(overflow_mask, inf_val, result)
        
        # Straight-Through Estimator (STE)
        result = x + (result - x).detach()
            
        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public interface: quantize the input tensor to the custom floating-point format.
        """
        sign, biased_exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, biased_exponent, significand, zero_mask, inf_mask, nan_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public interface (Compatible with nn.Module)
        """
        sign, biased_exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, biased_exponent, significand, zero_mask, inf_mask, nan_mask)