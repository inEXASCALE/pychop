import torch
# from .lightchop import LightChop
from .fixed_point import Chopf
from .integer import Chopi
import torch.nn as nn
from typing import Tuple

class QuantizedLayer(torch.nn.Module):
    """Example of a quantized linear layer"""
    def __init__(self, 
                 exp_bits: int,
                 sig_bits: int,
                 rmode: str = "nearest"):
        
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(x)


class IntQuantizedLayer(torch.nn.Module):
    """
    __init__(config)
        Apply ``pychop`` to quantization aware training, 
        One can feed [quant | chop | fixed_point] module as base for quantization.

    """

    def __init__(self, bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
        super(IntQuantizedLayer, self).__init__()
        self.chopi = Chopi(bits=bits, sign=sign, zpoint=zpoint, rd_func=rd_func, clip_range=clip_range, epsilon=epsilon)
        
    def forward(self, x):
        return self.chopi(x)
        

class FQuantizedLayer(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 ibits: int,
                 fbits: int,
                 rmode: str = "nearest",
                 bias: bool = True):
        """
        A linear layer with fixed-point quantization for weights, bias, and inputs.
            
        Parameters
        ----------
        in_dim : int
            Number of input features
        
        out_dim : int
            Number of output features
        
        ibits : int
            Number of integer bits (including sign) for Qm.n format
        
        fbits : int
            Number of fractional bits for Qm.n format
        
        rmode : int
            Rounding mode to use when quantizing the significand. Options are:
            - 0 or "nearest_odd": Round to nearest value, ties to odd.
            - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
            - 2 or "plus_inf": Round towards plus infinity (round up).
            - 3 or "minus_inf": Round towards minus infinity (round down).
            - 4 or "toward_zero": Truncate toward zero (no rounding up).
            - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
            - 6 or "stoc_equal": Stochastic rounding with 50% probability.

        bias : int
            Whether to include a bias term
        """
        super(FQuantizedLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quantizer = Chopf(ibits, fbits)
        self.rmode = rmode

        # Initialize weights and bias as floating-point parameters
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed-point quantization.
        
        Parameters
        ----------
        x : numpy.ndarray | jax.Array | torch.Tensor,
            The input tensor (batch_size, in_dim)

        Returns
        ----------
        Output: numpy.ndarray | jax.Array | torch.Tensor,
            The input tensor (batch_size, out_dim)
        """
        
        return self.quantizer.quantize(x, self.rmode)




class FPRound:
    def __init__(self, exp_bits: int, sig_bits: int, rmode: str = 1):
        """Initialize float precision simulator with custom format and rounding mode."""
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1  # Bias for IEEE 754-like format


    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to custom float representation with proper IEEE 754 handling."""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        exponent = torch.floor(torch.log2(abs_x.clamp(min=2.0**-24)))  # Minimum denormal
        significand = abs_x / (2.0 ** exponent)
        
        subnormal_mask = (exponent < self.min_exp)
        significand = torch.where(subnormal_mask, abs_x / (2.0 ** self.min_exp), significand)
        exponent = torch.where(subnormal_mask, self.min_exp, exponent)
        
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

        exp_min = 0  
        exp_max = 2 ** self.exp_bits - 1
        exponent = exponent.clamp(min=exp_min, max=exp_max)
        
        significand_steps = 2 ** self.sig_bits
        normal_mask = (exponent > 0) & (exponent < exp_max)
        subnormal_mask = (exponent == 0)
        significand_normal = significand - 1.0  
        
        if self.rmode == 1:
            significand_q = torch.round(significand_normal * significand_steps) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.round(significand * significand_steps) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 2:
            significand_q = torch.where(sign > 0, 
                                   torch.ceil(significand_normal * significand_steps),
                                   torch.floor(significand_normal * significand_steps)) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(sign > 0, 
                                             torch.ceil(significand * significand_steps), 
                                             torch.floor(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 3:
            significand_q = torch.where(sign > 0,
                                   torch.floor(significand_normal * significand_steps),
                                   torch.ceil(significand_normal * significand_steps)) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(sign > 0, 
                                             torch.floor(significand * significand_steps), 
                                             torch.ceil(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 4:
            significand_q = torch.floor(significand_normal * significand_steps) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.floor(significand * significand_steps) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 5:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            fraction = significand_scaled - floor_val
            prob = torch.rand_like(significand_scaled)
            significand_q = torch.where(prob < fraction, floor_val + 1, floor_val) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(torch.rand_like(significand) < (significand * significand_steps - torch.floor(significand * significand_steps)), 
                                             torch.ceil(significand * significand_steps), 
                                             torch.floor(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 6:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            prob = torch.rand_like(significand_scaled)
            significand_q = torch.where(prob < 0.5, floor_val, floor_val + 1) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(torch.rand_like(significand) < 0.5, 
                                             torch.floor(significand * significand_steps), 
                                             torch.ceil(significand * significand_steps)) / significand_steps, 
                                   significand_q)

        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")
        
        normal_result = sign * (1.0 + significand_q) * (2.0 ** (exponent - self.bias))
        subnormal_result = sign * significand_q * (2.0 ** self.min_exp)
        special_result = torch.where(inf_mask, torch.sign(x) * float('inf'), 
                                   torch.where(nan_mask, float('nan'), 0.0))
        
        result = torch.where(normal_mask, normal_result, 
                           torch.where(subnormal_mask, subnormal_result, 
                                     torch.where(zero_mask, 0.0, special_result)))
        
        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified precision using the initialized rounding mode."""
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask)
