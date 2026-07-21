"""JAX backend for BFP and Flexpoint quantization.

Array-level BFP quantization wraps the NumPy reference implementation and
returns JAX arrays. ``BFPQuantizerSTE`` adds a JAX custom VJP so gradients pass
through as a straight-through estimator. Flax is optional for array quantizers
and required only when instantiating ``BFPDense``.
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Any
import numpy as np

try:
    from flax import linen as nn
except ImportError:
    class _MissingFlax:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError("Flax is required for BFPDense")

        class initializers:
            @staticmethod
            def lecun_normal():
                raise ImportError("Flax is required for BFPDense")

            @staticmethod
            def zeros(*args, **kwargs):
                raise ImportError("Flax is required for BFPDense")

        @staticmethod
        def compact(fn):
            return fn

    nn = _MissingFlax()

# Import shared spec
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bfp_formats import BFPSpec, BFP_FORMATS, create_bfp_spec


# ============================================================================
# JAX Backend: BFP Tensor
# ============================================================================

class BFPTensor_:
    """JAX BFP tensor wrapper backed by the NumPy reference quantizer."""
    
    def __init__(
        self,
        data: jnp.ndarray,
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
        
        self.original_shape = data.shape
        
        # Use NumPy backend for quantization
        from ..np.bfp_formats import BFPTensor_ as NPBFPTensor
        data_np = np.array(data)
        self._np_impl = NPBFPTensor(data_np, format=self.spec)
    
    def dequantize(self) -> jnp.ndarray:
        """Dequantize to JAX array."""
        result_np = self._np_impl.dequantize()
        return jnp.array(result_np)
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        return self._np_impl.statistics()
    
    def __repr__(self):
        stats = self.statistics()
        return (f"BFPTensor_(backend=jax, shape={self.original_shape}, "
                f"format={self.spec.name})")


# ============================================================================
# JAX Backend: Custom VJP for STE
# ============================================================================

def create_bfp_ste_quantizer(spec: BFPSpec):
    """
    Create BFP quantizer with STE using JAX custom VJP.
    
    Parameters
    ----------
    spec : BFPSpec
        BFP format specification
    
    Returns
    -------
    callable
        Quantization function with STE
    """
    
    def quantize_fn(x):
        """Forward: apply BFP quantization."""
        bfp = BFPTensor_(x, format=spec)
        return bfp.dequantize()
    
    @jax.custom_vjp
    def quantize_with_ste(x):
        return quantize_fn(x)
    
    def quantize_fwd(x):
        """Forward pass."""
        return quantize_fn(x), (x,)
    
    def quantize_bwd(res, g):
        """Backward pass: STE (identity)."""
        x, = res
        return (g,)
    
    quantize_with_ste.defvjp(quantize_fwd, quantize_bwd)
    
    return quantize_with_ste


# ============================================================================
# BFP Quantizer for JAX/Flax
# ============================================================================

class BFPQuantizerSTE:
    """Callable JAX BFP quantizer with straight-through gradients.

    This class does not require Flax. It can be used directly on JAX arrays.
    """
    
    def __init__(self, format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'):
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
        
        # Create STE quantizer
        self._quantize_fn = create_bfp_ste_quantizer(self.spec)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply BFP quantization with STE."""
        return self._quantize_fn(x)
    
    def __repr__(self):
        return f"BFPQuantizerSTE(format={self.format_str})"


# ============================================================================
# Quantized Dense Layer for Flax
# ============================================================================

class BFPDense(nn.Module):
    """Flax dense layer with optional BFP input and weight quantization.
    
    Attributes
    ----------
    features : int
        Number of output features
    use_bias : bool
        Whether to use bias
    weight_format : str, BFPSpec, or tuple
        BFP format for weights
    quantize_input : bool
        Whether to quantize input
    """
    
    features: int
    use_bias: bool = True
    weight_format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'
    quantize_input: bool = True
    
    def setup(self):
        """Initialize quantizers."""
        self.weight_quantizer = BFPQuantizerSTE(format=self.weight_format)
        
        if self.quantize_input:
            self.input_quantizer = BFPQuantizerSTE(format=self.weight_format)
        else:
            self.input_quantizer = None
    
    @nn.compact
    def __call__(self, x):
        """Forward pass with BFP quantization."""
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
        
        # Add bias
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y = y + bias
        
        return y


# ============================================================================
# Convenience Function
# ============================================================================

def bfp_quantize(
    data: jnp.ndarray,
    format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'
) -> jnp.ndarray:
    """
    JAX backend: Quantize array to BFP format.
    """
    bfp_tensor = BFPTensor_(data, format=format)
    return bfp_tensor.dequantize()


__all__ = [
    'BFPTensor_',
    'BFPQuantizerSTE',
    'BFPDense',
    'create_bfp_ste_quantizer',
    'bfp_quantize',
]
