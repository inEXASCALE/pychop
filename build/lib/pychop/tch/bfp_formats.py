"""
Block Floating Point (BFP) - PyTorch Backend with STE

PyTorch implementation with Straight-Through Estimator for QAT.
Enables training neural networks with BFP quantization.

Author: Xinye Chen
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Union, Tuple, Optional, Any
import numpy as np

# Import shared spec
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bfp_formats import BFPSpec, BFP_FORMATS, create_bfp_spec


# ============================================================================
# PyTorch Backend: BFP Quantization with STE
# ============================================================================

class BFPQuantizeSTE(Function):
    """
    Straight-Through Estimator for BFP quantization.
    
    Forward: Real BFP quantization
    Backward: Gradient passes straight through (identity)
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, spec: BFPSpec) -> torch.Tensor:
        ctx.save_for_backward(input)
        
        # Use NumPy backend for quantization
        # (We'll optimize this later with pure PyTorch)
        input_np = input.detach().cpu().numpy()
        
        # Import NumPy implementation
        from ..np.bfp_formats import BFPTensor_
        bfp_tensor = BFPTensor_(input_np, format=spec)
        output_np = bfp_tensor.dequantize()
        
        # Convert back to torch
        output = torch.from_numpy(output_np).to(input.device).to(input.dtype)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # STE: pass gradient straight through
        return grad_output, None


# ============================================================================
# PyTorch Backend: BFP Tensor
# ============================================================================

class BFPTensor_:
    """
    PyTorch implementation of BFP tensor.
    
    Wraps NumPy implementation but returns PyTorch tensors.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'
    ):
        # Parse format
        if isinstance(format, str):
            if format.lower() not in BFP_FORMATS:
                raise ValueError(f"Unknown format: {format}")
            self.spec = BFP_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            mantissa_bits, block_size = format
            self.spec = create_bfp_spec(mantissa_bits, block_size)
        elif isinstance(format, BFPSpec):
            self.spec = format
        else:
            raise TypeError("format must be str, BFPSpec, or tuple")
        
        self.device = data.device
        self.dtype = data.dtype
        self.original_shape = data.shape
        
        # Use NumPy backend for quantization
        from ..np.bfp_formats import BFPTensor_  as NPBFPTensor
        data_np = data.detach().cpu().numpy()
        self._np_impl = NPBFPTensor(data_np, format=self.spec)
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize to PyTorch tensor."""
        result_np = self._np_impl.dequantize()
        return torch.from_numpy(result_np).to(self.device).to(self.dtype)
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        return self._np_impl.statistics()
    
    def __repr__(self):
        stats = self.statistics()
        return (f"BFPTensor_(backend=torch, shape={self.original_shape}, "
                f"format={self.spec.name}, device={self.device})")


# ============================================================================
# BFP Quantizer Module with STE
# ============================================================================

class BFPQuantizerSTE(nn.Module):
    """
    BFP quantizer with STE for QAT.
    
    Automatically uses STE during training (requires_grad=True).
    """
    
    def __init__(self, format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'):
        super().__init__()
        
        # Parse format
        if isinstance(format, str):
            if format.lower() not in BFP_FORMATS:
                raise ValueError(f"Unknown format: {format}")
            self.spec = BFP_FORMATS[format.lower()]
            self.format_str = format
        elif isinstance(format, tuple):
            mantissa_bits, block_size = format
            self.spec = create_bfp_spec(mantissa_bits, block_size)
            self.format_str = f"BFP{mantissa_bits}"
        elif isinstance(format, BFPSpec):
            self.spec = format
            self.format_str = format.name
        else:
            raise TypeError("format must be str, BFPSpec, or tuple")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BFP quantization with STE."""
        if x.requires_grad and torch.is_grad_enabled():
            # Training: use STE
            return BFPQuantizeSTE.apply(x, self.spec)
        else:
            # Inference: no autograd
            with torch.no_grad():
                return BFPQuantizeSTE.apply(x, self.spec)
    
    def __repr__(self):
        return f"BFPQuantizerSTE(format={self.format_str})"


# ============================================================================
# Quantized Linear Layer
# ============================================================================

class BFPLinear(nn.Linear):
    """
    Linear layer with BFP quantization.
    
    Quantizes weights and optionally activations.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8',
        act_format: Optional[Union[str, BFPSpec, Tuple[int, int]]] = None,
        quantize_input: bool = True,
        quantize_output: bool = False,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        
        self.weight_quantizer = BFPQuantizerSTE(format=weight_format)
        
        if quantize_input:
            act_fmt = act_format if act_format is not None else weight_format
            self.input_quantizer = BFPQuantizerSTE(format=act_fmt)
        else:
            self.input_quantizer = None
        
        if quantize_output:
            act_fmt = act_format if act_format is not None else weight_format
            self.output_quantizer = BFPQuantizerSTE(format=act_fmt)
        else:
            self.output_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        
        weight_q = self.weight_quantizer(self.weight)
        output = torch.nn.functional.linear(x, weight_q, self.bias)
        
        if self.output_quantizer is not None:
            output = self.output_quantizer(output)
        
        return output


# ============================================================================
# Model Conversion Utility
# ============================================================================

def convert_linear_to_bfp(
    module: nn.Module,
    format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8',
    quantize_input: bool = True,
    quantize_output: bool = False,
    inplace: bool = True
) -> nn.Module:
    """
    Convert all Linear layers to BFP quantized versions.
    """
    import copy
    
    if not inplace:
        module = copy.deepcopy(module)
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, BFPLinear):
            bfp_layer = BFPLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                weight_format=format,
                quantize_input=quantize_input,
                quantize_output=quantize_output
            )
            
            bfp_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                bfp_layer.bias.data = child.bias.data.clone()
            
            setattr(module, name, bfp_layer)
        else:
            convert_linear_to_bfp(child, format, quantize_input, quantize_output, inplace=True)
    
    return module


# ============================================================================
# Convenience Function
# ============================================================================

def bfp_quantize(
    data: torch.Tensor,
    format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'
) -> torch.Tensor:
    """
    PyTorch backend: Quantize tensor to BFP format.
    
    Automatically uses STE if requires_grad=True.
    """
    if data.requires_grad and torch.is_grad_enabled():
        # Use STE
        quantizer = BFPQuantizerSTE(format=format)
        return quantizer(data)
    else:
        # Direct quantization
        bfp_tensor = BFPTensor_(data, format=format)
        return bfp_tensor.dequantize()


__all__ = [
    'BFPTensor_',
    'BFPQuantizeSTE',
    'BFPQuantizerSTE',
    'BFPLinear',
    'convert_linear_to_bfp',
    'bfp_quantize',
]