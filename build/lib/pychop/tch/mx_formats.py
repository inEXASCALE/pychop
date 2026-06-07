"""
Microscaling (MX) Formats - PyTorch Backend with STE

PyTorch implementation with Straight-Through Estimator for QAT.
Enables training neural networks with MX quantization.

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
from mx_formats import MXSpec, MX_FORMATS, create_mx_spec


# ============================================================================
# PyTorch Backend: MX Quantization with STE
# ============================================================================

class MXQuantizeSTE(Function):
    """
    Straight-Through Estimator for MX quantization.
    
    Forward: Real MX quantization
    Backward: Gradient passes straight through (identity)
    """
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        spec: MXSpec,
        block_size: int,
        scale_exp_bits: Optional[int],
        scale_sig_bits: Optional[int]
    ) -> torch.Tensor:
        ctx.save_for_backward(input)
        
        # Use NumPy backend for quantization
        input_np = input.detach().cpu().numpy()
        
        from ..np.mx_formats import MXTensor_
        mx_tensor = MXTensor_(
            input_np,
            format=spec,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits
        )
        output_np = mx_tensor.dequantize()
        
        # Convert back to torch
        output = torch.from_numpy(output_np).to(input.device).to(input.dtype)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        # STE: pass gradient straight through
        return grad_output, None, None, None, None


# ============================================================================
# PyTorch Backend: MX Tensor
# ============================================================================

class MXTensor_:
    """
    PyTorch implementation of MX tensor.
    
    Wraps NumPy implementation but returns PyTorch tensors.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        block_size: int = 32,
        scale_exp_bits: Optional[int] = None,
        scale_sig_bits: Optional[int] = None
    ):
        # Parse format
        if isinstance(format, str):
            if format.lower() not in MX_FORMATS:
                raise ValueError(f"Unknown format: {format}")
            self.spec = MX_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            exp_bits, sig_bits = format
            self.spec = create_mx_spec(exp_bits, sig_bits, block_size)
        elif isinstance(format, MXSpec):
            self.spec = format
        else:
            raise TypeError("format must be str, MXSpec, or tuple")
        
        self.device = data.device
        self.dtype = data.dtype
        self.original_shape = data.shape
        self.scale_exp_bits = scale_exp_bits
        self.scale_sig_bits = scale_sig_bits
        
        # Use NumPy backend
        from ..np.mx_formats import MXTensor_ as NPMXTensor
        data_np = data.detach().cpu().numpy()
        self._np_impl = NPMXTensor(
            data_np,
            format=self.spec,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits
        )
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize to PyTorch tensor."""
        result_np = self._np_impl.dequantize()
        return torch.from_numpy(result_np).to(self.device).to(self.dtype)
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        return self._np_impl.statistics()
    
    def __repr__(self):
        stats = self.statistics()
        return (f"MXTensor_(backend=torch, shape={self.original_shape}, "
                f"format={self.spec.name}, device={self.device})")


# ============================================================================
# MX Quantizer Module with STE
# ============================================================================

class MXQuantizerSTE(nn.Module):
    """
    MX quantizer with STE for QAT.
    
    Automatically uses STE during training.
    """
    
    def __init__(
        self,
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        block_size: int = 32,
        scale_exp_bits: Optional[int] = None,
        scale_sig_bits: Optional[int] = None
    ):
        super().__init__()
        
        # Parse format
        if isinstance(format, str):
            if format.lower() not in MX_FORMATS:
                raise ValueError(f"Unknown format: {format}")
            self.spec = MX_FORMATS[format.lower()]
            self.format_str = format
        elif isinstance(format, tuple):
            exp_bits, sig_bits = format
            self.spec = create_mx_spec(exp_bits, sig_bits, block_size)
            self.format_str = f"E{exp_bits}M{sig_bits}"
        elif isinstance(format, MXSpec):
            self.spec = format
            self.format_str = format.name
        else:
            raise TypeError("format must be str, MXSpec, or tuple")
        
        self.block_size = block_size
        self.scale_exp_bits = scale_exp_bits
        self.scale_sig_bits = scale_sig_bits
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MX quantization with STE."""
        if x.requires_grad and torch.is_grad_enabled():
            # Training: use STE
            return MXQuantizeSTE.apply(
                x, self.spec, self.block_size,
                self.scale_exp_bits, self.scale_sig_bits
            )
        else:
            # Inference
            with torch.no_grad():
                return MXQuantizeSTE.apply(
                    x, self.spec, self.block_size,
                    self.scale_exp_bits, self.scale_sig_bits
                )
    
    def __repr__(self):
        return f"MXQuantizerSTE(format={self.format_str}, block_size={self.block_size})"


# ============================================================================
# Quantized Linear Layer
# ============================================================================

class MXLinear(nn.Linear):
    """
    Linear layer with MX quantization.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        act_format: Optional[Union[str, MXSpec, Tuple[int, int]]] = None,
        block_size: int = 32,
        quantize_input: bool = True,
        quantize_output: bool = False,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        
        self.weight_quantizer = MXQuantizerSTE(
            format=weight_format,
            block_size=block_size
        )
        
        if quantize_input:
            act_fmt = act_format if act_format is not None else weight_format
            self.input_quantizer = MXQuantizerSTE(
                format=act_fmt,
                block_size=block_size
            )
        else:
            self.input_quantizer = None
        
        if quantize_output:
            act_fmt = act_format if act_format is not None else weight_format
            self.output_quantizer = MXQuantizerSTE(
                format=act_fmt,
                block_size=block_size
            )
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
# Quantized Attention Layer
# ============================================================================

class MXAttention(nn.Module):
    """
    Multi-head attention with MX quantization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        block_size: int = 32,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # Q, K, V projections
        self.q_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                               weight_format=format, block_size=block_size)
        self.k_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                               weight_format=format, block_size=block_size)
        self.v_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                               weight_format=format, block_size=block_size)
        self.out_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                                 weight_format=format, block_size=block_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Quantizer for attention scores
        self.score_quantizer = MXQuantizerSTE(format=format, block_size=block_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size = query.size(0)
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = self.score_quantizer(scores)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


# ============================================================================
# Model Conversion Utility
# ============================================================================

def convert_linear_to_mx(
    module: nn.Module,
    format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
    block_size: int = 32,
    quantize_input: bool = True,
    quantize_output: bool = False,
    inplace: bool = True
) -> nn.Module:
    """
    Convert all Linear layers to MX quantized versions.
    """
    import copy
    
    if not inplace:
        module = copy.deepcopy(module)
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, MXLinear):
            mx_layer = MXLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                weight_format=format,
                block_size=block_size,
                quantize_input=quantize_input,
                quantize_output=quantize_output
            )
            
            mx_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                mx_layer.bias.data = child.bias.data.clone()
            
            setattr(module, name, mx_layer)
        else:
            convert_linear_to_mx(
                child, format, block_size,
                quantize_input, quantize_output, inplace=True
            )
    
    return module


# ============================================================================
# Convenience Function
# ============================================================================

def mx_quantize(
    data: torch.Tensor,
    format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
    block_size: int = 32,
    scale_exp_bits: Optional[int] = None,
    scale_sig_bits: Optional[int] = None
) -> torch.Tensor:
    """
    PyTorch backend: Quantize tensor to MX format.
    """
    if data.requires_grad and torch.is_grad_enabled():
        quantizer = MXQuantizerSTE(
            format=format,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits
        )
        return quantizer(data)
    else:
        mx_tensor = MXTensor_(
            data,
            format=format,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits
        )
        return mx_tensor.dequantize()


def compare_mx_formats(
    data: torch.Tensor,
    formats: Optional[list] = None,
    block_size: int = 32
) -> None:
    """Compare different MX formats (PyTorch backend)."""
    data_np = data.detach().cpu().numpy()
    from ..np.mx_formats import compare_mx_formats as np_compare
    np_compare(data_np, formats=formats, block_size=block_size)


__all__ = [
    'MXTensor_',
    'MXQuantizeSTE',
    'MXQuantizerSTE',
    'MXLinear',
    'MXAttention',
    'convert_linear_to_mx',
    'mx_quantize',
    'compare_mx_formats',
]