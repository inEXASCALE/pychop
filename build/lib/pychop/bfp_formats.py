"""
Block Floating Point (BFP) Format - Backend Agnostic Entry Point

This module provides automatic backend detection and routing for BFP quantization.
Supports NumPy, JAX, and PyTorch backends with automatic detection.

Usage:
    >>> import pychop
    >>> pychop.backend('auto')  # Auto-detect from input
    >>> 
    >>> # NumPy
    >>> import numpy as np
    >>> X = np.random.randn(1024, 768)
    >>> X_q = bfp_quantize(X, format='bfp8')
    >>> 
    >>> # PyTorch (with STE for training)
    >>> import torch
    >>> X = torch.randn(128, 768, requires_grad=True)
    >>> X_q = bfp_quantize(X, format='bfp8')  # Automatic STE!
    >>> 
    >>> # JAX
    >>> import jax.numpy as jnp
    >>> X = jnp.array(np.random.randn(512, 512))
    >>> X_q = bfp_quantize(X, format='bfp8')

Author: Xinye Chen

"""

import os
from typing import Union, Tuple, Optional, Any
from dataclasses import dataclass


# ============================================================================
# Backend Detection (inline to avoid import issues)
# ============================================================================

def _detect_array_type(x: Any) -> str:
    """
    Detect backend from input array type.
    
    Parameters
    ----------
    x : Any
        Input array or scalar
    
    Returns
    -------
    str
        Backend name: 'numpy', 'torch', or 'jax'
    """
    module = type(x).__module__
    
    if "torch" in module:
        return "torch"
    if "jax" in module:
        return "jax"
    return "numpy"


def _get_backend_env() -> str:
    """Get backend from environment variable."""
    return os.environ.get('chop_backend', 'auto')


# ============================================================================
# BFP Format Specification (Backend-Independent)
# ============================================================================

@dataclass
class BFPSpec:
    """
    Block Floating Point format specification.
    
    This is backend-independent and shared across all implementations.
    
    Attributes
    ----------
    name : str
        Format name
    mantissa_bits : int
        Number of mantissa bits per element (including sign)
    block_size : int
        Number of elements sharing same exponent
    exponent_bits : int
        Number of bits for shared exponent
    has_sign : bool
        Whether elements have sign bits
    use_subnormals : bool
        Whether to support subnormal numbers
    """
    name: str
    mantissa_bits: int
    block_size: int
    exponent_bits: int = 8
    has_sign: bool = True
    use_subnormals: bool = False
    
    @property
    def total_bits_per_block(self) -> int:
        """Total bits for entire block."""
        return self.exponent_bits + (self.mantissa_bits * self.block_size)
    
    @property
    def compression_vs_fp32(self) -> float:
        """Compression ratio vs FP32."""
        fp32_bits = 32 * self.block_size
        return fp32_bits / self.total_bits_per_block
    
    @property
    def compression_vs_fp16(self) -> float:
        """Compression ratio vs FP16."""
        fp16_bits = 16 * self.block_size
        return fp16_bits / self.total_bits_per_block
    
    def __repr__(self):
        return (f"BFPSpec(name='{self.name}', mantissa={self.mantissa_bits}b, "
                f"block_size={self.block_size}, exponent={self.exponent_bits}b)")


# Predefined BFP formats (shared across all backends)
BFP_FORMATS = {
    'bfp16': BFPSpec('bfp16', mantissa_bits=16, block_size=16, exponent_bits=8),
    'bfp12': BFPSpec('bfp12', mantissa_bits=12, block_size=16, exponent_bits=8),
    'bfp8': BFPSpec('bfp8', mantissa_bits=8, block_size=32, exponent_bits=8),
    'bfp6': BFPSpec('bfp6', mantissa_bits=6, block_size=32, exponent_bits=8),
    'bfp4': BFPSpec('bfp4', mantissa_bits=4, block_size=32, exponent_bits=8),
    'bfp3': BFPSpec('bfp3', mantissa_bits=3, block_size=64, exponent_bits=8),
    'bfp2': BFPSpec('bfp2', mantissa_bits=2, block_size=128, exponent_bits=8),
    'flexpoint16': BFPSpec('flexpoint16', mantissa_bits=16, block_size=16, exponent_bits=5),
    'flexpoint8': BFPSpec('flexpoint8', mantissa_bits=8, block_size=32, exponent_bits=5),
}


def create_bfp_spec(
    mantissa_bits: int,
    block_size: int,
    exponent_bits: int = 8,
    name: Optional[str] = None
) -> BFPSpec:
    """
    Create custom BFP format specification.
    
    Parameters
    ----------
    mantissa_bits : int
        Number of mantissa bits (1-32)
    block_size : int
        Elements per block
    exponent_bits : int
        Bits for shared exponent
    name : str, optional
        Custom name
    
    Returns
    -------
    BFPSpec
        BFP format specification
    """
    if name is None:
        name = f"custom_bfp{mantissa_bits}"
    
    return BFPSpec(
        name=name,
        mantissa_bits=mantissa_bits,
        block_size=block_size,
        exponent_bits=exponent_bits
    )


# ============================================================================
# Backend Detection and Routing
# ============================================================================

def _resolve_backend(X: Any = None) -> str:
    """
    Resolve which backend to use.
    
    Parameters
    ----------
    X : Any, optional
        Input array (if provided, used for auto-detection)
    
    Returns
    -------
    str
        Backend name: 'numpy', 'jax', or 'torch'
    """
    env_backend = _get_backend_env()
    
    if env_backend == 'auto':
        if X is not None:
            return _detect_array_type(X)
        else:
            # Default to numpy if no input provided
            return 'numpy'
    
    if env_backend not in {'numpy', 'jax', 'torch'}:
        raise ValueError(
            f"Invalid backend: {env_backend}. "
            "Must be 'numpy', 'jax', 'torch', or 'auto'."
        )
    
    return env_backend


def _get_backend_module(backend: str):
    """
    Get backend-specific BFP implementation.
    
    Parameters
    ----------
    backend : str
        Backend name
    
    Returns
    -------
    module
        Backend-specific BFP module
    """
    if backend == 'torch':
        try:
            from .tch import bfp_formats as backend_module
        except ImportError:
            raise ImportError(
                "PyTorch backend not available. "
                "Install with: pip install torch"
            )
    elif backend == 'jax':
        try:
            from .jx import bfp_formats as backend_module
        except ImportError:
            raise ImportError(
                "JAX backend not available. "
                "Install with: pip install jax jaxlib flax"
            )
    elif backend == 'numpy':
        from .np import bfp_formats as backend_module
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return backend_module


# ============================================================================
# User-Facing Functions
# ============================================================================

def bfp_quantize(
    data: Any,
    format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8',
    backend: Optional[str] = None
) -> Any:
    """
    Quantize array to BFP format.
    
    Automatically detects backend from input type or uses specified backend.
    
    Parameters
    ----------
    data : array-like
        Input data (numpy.ndarray, torch.Tensor, or jax.Array)
    format : str, BFPSpec, or tuple
        BFP format specification
    backend : str, optional
        Force specific backend ('numpy', 'jax', or 'torch')
        If None, auto-detects from input
    
    Returns
    -------
    array-like
        Quantized data (same type as input)
    
    Examples
    --------
    >>> # NumPy
    >>> import numpy as np
    >>> X = np.random.randn(1024, 768)
    >>> X_q = bfp_quantize(X, format='bfp8')
    >>> 
    >>> # PyTorch (with automatic STE if requires_grad=True)
    >>> import torch
    >>> X = torch.randn(128, 768, requires_grad=True)
    >>> X_q = bfp_quantize(X, format='bfp8')
    >>> loss = X_q.sum()
    >>> loss.backward()  # Gradients flow through!
    >>> 
    >>> # Custom format
    >>> X_q = bfp_quantize(X, format=(4, 32))  # 4-bit mantissa, 32 elem/block
    """
    # Resolve backend
    if backend is None:
        backend = _resolve_backend(data)
    
    # Get backend module
    backend_module = _get_backend_module(backend)
    
    # Call backend-specific quantization
    return backend_module.bfp_quantize(data, format=format)


class BFPTensor:
    """
    Backend-agnostic BFP tensor wrapper.
    
    Automatically routes to appropriate backend implementation.
    
    Parameters
    ----------
    data : array-like
        Input tensor
    format : str, BFPSpec, or tuple
        BFP format
    backend : str, optional
        Force specific backend
    
    Examples
    --------
    >>> # NumPy backend
    >>> import numpy as np
    >>> X = np.random.randn(1024, 768)
    >>> bfp = BFPTensor(X, format='bfp8')
    >>> X_reconstructed = bfp.dequantize()
    >>> stats = bfp.statistics()
    """
    
    def __init__(
        self,
        data: Any,
        format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8',
        backend: Optional[str] = None
    ):
        # Resolve backend
        if backend is None:
            self.backend = _resolve_backend(data)
        else:
            self.backend = backend
        
        # Get backend module
        backend_module = _get_backend_module(self.backend)
        
        # Create backend-specific tensor
        self._impl = backend_module.BFPTensor_(data, format=format)
    
    def dequantize(self) -> Any:
        """Dequantize to original data type."""
        return self._impl.dequantize()
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        return self._impl.statistics()
    
    def __repr__(self):
        return f"BFPTensor(backend={self.backend}, impl={self._impl})"


def print_bfp_format_table():
    """Print table of predefined BFP formats."""
    print("="*90)
    print("Predefined BFP Formats")
    print("="*90)
    
    header = (f"{'Name':<15} {'Mantissa':<10} {'Block Size':<12} "
              f"{'Exponent':<10} {'Compress FP16':<15} {'Total Bits':<12}")
    print(header)
    print("-"*90)
    
    for name, spec in BFP_FORMATS.items():
        row = (f"{spec.name:<15} "
               f"{spec.mantissa_bits:<10} "
               f"{spec.block_size:<12} "
               f"{spec.exponent_bits:<10} "
               f"{spec.compression_vs_fp16:.2f}x{'':>11} "
               f"{spec.total_bits_per_block}")
        print(row)
    
    print("="*90)




__all__ = [
    'BFPSpec',
    'BFPTensor',
    'BFP_FORMATS',
    'create_bfp_spec',
    'bfp_quantize',
    'print_bfp_format_table',
]