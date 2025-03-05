import torch
from typing import Tuple

class Rounding:
    """
    A class to simulate different floating-point precisions and rounding modes
    for PyTorch tensors.
    """
    def __init__(self, exp_bits: int, sig_bits: int):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1  # Bias for IEEE 754
        
    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize with specific format parameters.
        Convert to custom float representation with proper IEEE 754 handling
        
        Args:
            exp_bits: Number of bits for exponent
            sig_bits: Number of bits for significand (significant digits)
        """
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
                           rounding_mode: str) -> torch.Tensor:
        """Quantize components according to IEEE 754 FP16 rules with various rounding modes"""
        
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
        if rounding_mode == "nearest":
            significand_q = torch.round(significand_normal * significand_steps) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.round(significand[subnormal_mask] * 
                                                       significand_steps) / significand_steps
        elif rounding_mode == "up":
            significand_q = torch.where(sign > 0, 
                                   torch.ceil(significand_normal * significand_steps),
                                   torch.floor(significand_normal * significand_steps)) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.where(sign[subnormal_mask] > 0,
                                                       torch.ceil(significand[subnormal_mask] * significand_steps),
                                                       torch.floor(significand[subnormal_mask] * significand_steps)) / significand_steps
        elif rounding_mode == "down":
            significand_q = torch.where(sign > 0,
                                   torch.floor(significand_normal * significand_steps),
                                   torch.ceil(significand_normal * significand_steps)) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.where(sign[subnormal_mask] > 0,
                                                       torch.floor(significand[subnormal_mask] * significand_steps),
                                                       torch.ceil(significand[subnormal_mask] * significand_steps)) / significand_steps
        elif rounding_mode == "towards_zero":
            significand_q = torch.trunc(significand_normal * significand_steps) / significand_steps
            if torch.any(subnormal_mask):
                significand_q[subnormal_mask] = torch.trunc(significand[subnormal_mask] * 
                                                       significand_steps) / significand_steps
        elif rounding_mode == "stochastic_equal":
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
        elif rounding_mode == "stochastic_proportional":
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
        else:
            raise ValueError(f"Unsupported rounding mode: {rounding_mode}")
        
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

    def quantize(self, x: torch.Tensor, rounding_mode: str = "nearest") -> torch.Tensor:
        """
        Quantize a tensor to the specified precision with given rounding mode.
        
        Args:
            x: Input tensor
            rounding_mode: One of 'nearest', 'up', 'down', 'towards_zero', 
                          'stochastic_equal', 'stochastic_proportional'
        """
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, rounding_mode)


class QuantizedLayer(torch.nn.Module):
    """Example of a quantized linear layer"""
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 exp_bits: int,
                 sig_bits: int,
                 rounding_mode: str = "nearest"):
        
        super().__init__()
        self.quantizer = Rounding(exp_bits, sig_bits)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.rounding_mode = rounding_mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights and input
        q_weight = self.quantizer.quantize(self.weight, self.rounding_mode)
        q_input = self.quantizer.quantize(x, self.rounding_mode)
        
        # Perform computation
        output = torch.matmul(q_input, q_weight.t())
        
        # Quantize bias and add
        q_bias = self.quantizer.quantize(self.bias, self.rounding_mode)
        output = output + q_bias
        
        return output
