"""
Microscaling (MX) Formats - JAX Backend with Custom VJP

JAX implementation with custom VJP for differentiation.
Enables training with MX quantization in JAX/Flax.

Author: Xinye Chen
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Union, Tuple, Optional, Any
import numpy as np

# Import shared spec
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mx_formats import MXSpec, MX_FORMATS, create_mx_spec


# ============================================================================
# JAX Backend: MX Tensor
# ============================================================================

class MXTensor_:
    """
    JAX implementation of MX tensor.
    """
    
    def __init__(
        self,
        data: jnp.ndarray,
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
        
        self.original_shape = data.shape
        self.scale_exp_bits = scale_exp_bits
        self.scale_sig_bits = scale_sig_bits
        
        # Use NumPy backend
        from ..np.mx_formats import MXTensor_ as NPMXTensor
        data_np = np.array(data)
        self._np_impl = NPMXTensor(
            data_np,
            format=self.spec,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits
        )
    
    def dequantize(self) -> jnp.ndarray:
        """Dequantize to JAX array."""
        result_np = self._np_impl.dequantize()
        return jnp.array(result_np)
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        return self._np_impl.statistics()
    
    def __repr__(self):
        stats = self.statistics()
        return (f"MXTensor_(backend=jax, shape={self.original_shape}, "
                f"format={self.spec.name})")


# ============================================================================
# JAX Backend: Custom VJP for STE
# ============================================================================

def create_mx_ste_quantizer(
    spec: MXSpec,
    block_size: int = 32,
    scale_exp_bits: Optional[int] = None,
    scale_sig_bits: Optional[int] = None
):
    """
    Create MX quantizer with STE using JAX custom VJP.
    """
    
    def quantize_fn(x):
        """Forward: apply MX quantization."""
        mx = MXTensor_(x, format=spec, block_size=block_size,
                      scale_exp_bits=scale_exp_bits,
                      scale_sig_bits=scale_sig_bits)
        return mx.dequantize()
    
    @jax.custom_vjp
    def quantize_with_ste(x):
        return quantize_fn(x)
    
    def quantize_fwd(x):
        return quantize_fn(x), (x,)
    
    def quantize_bwd(res, g):
        x, = res
        return (g,)
    
    quantize_with_ste.defvjp(quantize_fwd, quantize_bwd)
    
    return quantize_with_ste


# ============================================================================
# MX Quantizer for JAX/Flax
# ============================================================================

class MXQuantizerSTE:
    """
    MX quantizer with STE for JAX/Flax.
    """
    
    def __init__(
        self,
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
        
        # Create STE quantizer
        self._quantize_fn = create_mx_ste_quantizer(
            self.spec, block_size, scale_exp_bits, scale_sig_bits
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply MX quantization with STE."""
        return self._quantize_fn(x)
    
    def __repr__(self):
        return f"MXQuantizerSTE(format={self.format_str}, block_size={self.block_size})"


# ============================================================================
# Quantized Dense Layer for Flax
# ============================================================================

class MXDense(nn.Module):
    """
    Dense layer with MX quantization for Flax.
    """
    
    features: int
    use_bias: bool = True
    weight_format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3'
    block_size: int = 32
    quantize_input: bool = True
    
    def setup(self):
        self.weight_quantizer = MXQuantizerSTE(
            format=self.weight_format,
            block_size=self.block_size
        )
        
        if self.quantize_input:
            self.input_quantizer = MXQuantizerSTE(
                format=self.weight_format,
                block_size=self.block_size
            )
        else:
            self.input_quantizer = None
    
    @nn.compact
    def __call__(self, x):
        # Quantize input
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        
        # Create and quantize kernel
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.features)
        )
        kernel_q = self.weight_quantizer(kernel)
        
        # Linear operation
        y = jnp.dot(x, kernel_q)
        
        # Bias
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y = y + bias
        
        return y


# ============================================================================
# Convenience Function
# ============================================================================

def mx_quantize(
    data: jnp.ndarray,
    format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
    block_size: int = 32,
    scale_exp_bits: Optional[int] = None,
    scale_sig_bits: Optional[int] = None
) -> jnp.ndarray:
    """
    JAX backend: Quantize array to MX format.
    """
    mx_tensor = MXTensor_(
        data, format, block_size,
        scale_exp_bits, scale_sig_bits
    )
    return mx_tensor.dequantize()


def compare_mx_formats(
    data: jnp.ndarray,
    formats: Optional[list] = None,
    block_size: int = 32
) -> None:
    """Compare different MX formats (JAX backend)."""
    data_np = np.array(data)
    from ..np.mx_formats import compare_mx_formats as np_compare
    np_compare(data_np, formats=formats, block_size=block_size)


__all__ = [
    'MXTensor_',
    'MXQuantizerSTE',
    'MXDense',
    'create_mx_ste_quantizer',
    'mx_quantize',
    'compare_mx_formats',
]