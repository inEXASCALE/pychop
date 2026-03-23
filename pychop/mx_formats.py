"""
Microscaling (MX) Formats - Backend Agnostic Entry Point

OCP Microscaling format with automatic backend detection.
Supports NumPy, JAX, and PyTorch backends.

MX Format Structure:
- Block of N elements (typically 32)
- One shared scale factor (exponent) per block
- Each element has its own exponent and mantissa
- Significantly better dynamic range than BFP

Reference:
    OCP Microscaling Formats (MX) v1.0 Specification
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Usage:
    >>> import pychop
    >>> pychop.backend('auto')  # Auto-detect
    >>> 
    >>> # NumPy
    >>> import numpy as np
    >>> X = np.random.randn(1024, 768)
    >>> X_q = mx_quantize(X, format='mxfp8_e4m3')
    >>> 
    >>> # PyTorch (with STE)
    >>> import torch
    >>> X = torch.randn(128, 768, requires_grad=True)
    >>> X_q = mx_quantize(X, format='mxfp8_e4m3')
    >>> loss = X_q.sum()
    >>> loss.backward()  # Automatic STE!
    >>> 
    >>> # JAX
    >>> import jax.numpy as jnp
    >>> X = jnp.array(np.random.randn(512, 512))
    >>> X_q = mx_quantize(X, format='mxfp8_e4m3')

Author: Xinye Chen
"""

import os
from typing import Union, Tuple, Optional, Any, Dict
from dataclasses import dataclass


# ============================================================================
# Backend Detection (inline to avoid import issues)
# ============================================================================

def _detect_array_type(x: Any) -> str:
    """Detect backend from input array type."""
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
# MX Format Specification (Backend-Independent)
# ============================================================================

@dataclass
class MXSpec:
    """
    Microscaling format specification.
    
    MX format uses:
    - Shared scale (exponent) for block of elements
    - Individual exponent + mantissa for each element
    
    Attributes
    ----------
    name : str
        Format name (e.g., 'MXFP8_E4M3')
    exp_bits : int
        Element exponent bits
    sig_bits : int
        Element significand bits (excluding implicit 1)
    block_size : int
        Elements per block
    scale_exp_bits : int
        Scale factor exponent bits
    scale_sig_bits : int
        Scale factor significand bits
    """
    name: str
    exp_bits: int
    sig_bits: int
    block_size: int = 32
    scale_exp_bits: int = 8
    scale_sig_bits: int = 0  # Scale is typically just exponent
    
    @property
    def element_bits(self) -> int:
        """Total bits per element (1 sign + exp + sig)."""
        return 1 + self.exp_bits + self.sig_bits
    
    @property
    def total_bits_per_block(self) -> int:
        """Total bits for entire block (elements + scale)."""
        element_bits = self.element_bits * self.block_size
        scale_bits = self.scale_exp_bits + self.scale_sig_bits
        return element_bits + scale_bits
    
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
        return (f"MXSpec(name='{self.name}', E{self.exp_bits}M{self.sig_bits}, "
                f"block_size={self.block_size})")


# Predefined MX formats (OCP standard)
MX_FORMATS = {
    # MXFP8 formats
    'mxfp8_e5m2': MXSpec('MXFP8_E5M2', exp_bits=5, sig_bits=2, block_size=32),
    'mxfp8_e4m3': MXSpec('MXFP8_E4M3', exp_bits=4, sig_bits=3, block_size=32),
    
    # MXFP6 formats
    'mxfp6_e3m2': MXSpec('MXFP6_E3M2', exp_bits=3, sig_bits=2, block_size=32),
    'mxfp6_e2m3': MXSpec('MXFP6_E2M3', exp_bits=2, sig_bits=3, block_size=32),
    
    # MXFP4 format
    'mxfp4_e2m1': MXSpec('MXFP4_E2M1', exp_bits=2, sig_bits=1, block_size=32),
    
    # MXINT8 (integer format with MX scaling)
    'mxint8': MXSpec('MXINT8', exp_bits=0, sig_bits=7, block_size=32),
}


def create_mx_spec(
    exp_bits: int,
    sig_bits: int,
    block_size: int = 32,
    scale_exp_bits: int = 8,
    name: Optional[str] = None
) -> MXSpec:
    """Create custom MX format specification."""
    if name is None:
        total_bits = 1 + exp_bits + sig_bits
        name = f"CUSTOM_MX{total_bits}_E{exp_bits}M{sig_bits}"
    
    return MXSpec(
        name=name,
        exp_bits=exp_bits,
        sig_bits=sig_bits,
        block_size=block_size,
        scale_exp_bits=scale_exp_bits
    )


# ============================================================================
# Backend Detection and Routing
# ============================================================================

def _resolve_backend(X: Any = None) -> str:
    """Resolve which backend to use."""
    env_backend = _get_backend_env()
    
    if env_backend == 'auto':
        if X is not None:
            return _detect_array_type(X)
        else:
            return 'numpy'
    
    if env_backend not in {'numpy', 'jax', 'torch'}:
        raise ValueError(
            f"Invalid backend: {env_backend}. "
            "Must be 'numpy', 'jax', 'torch', or 'auto'."
        )
    
    return env_backend


def _get_backend_module(backend: str):
    """Get backend-specific MX implementation."""
    if backend == 'torch':
        try:
            from .tch import mx_formats as backend_module
        except ImportError:
            raise ImportError(
                "PyTorch backend not available. "
                "Install with: pip install torch"
            )
    elif backend == 'jax':
        try:
            from .jx import mx_formats as backend_module
        except ImportError:
            raise ImportError(
                "JAX backend not available. "
                "Install with: pip install jax jaxlib flax"
            )
    elif backend == 'numpy':
        from .np import mx_formats as backend_module
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return backend_module


# ============================================================================
# User-Facing Functions
# ============================================================================

def mx_quantize(
    data: Any,
    format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
    block_size: int = 32,
    scale_exp_bits: Optional[int] = None,
    scale_sig_bits: Optional[int] = None,
    backend: Optional[str] = None
) -> Any:
    """
    Quantize array to MX format.
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(1024, 768)
    >>> X_q = mx_quantize(X, format='mxfp8_e4m3')
    """
    # Resolve backend
    if backend is None:
        backend = _resolve_backend(data)
    
    # Get backend module
    backend_module = _get_backend_module(backend)
    
    # Call backend-specific quantization
    return backend_module.mx_quantize(
        data,
        format=format,
        block_size=block_size,
        scale_exp_bits=scale_exp_bits,
        scale_sig_bits=scale_sig_bits
    )


class MXTensor:
    """Backend-agnostic MX tensor wrapper."""
    
    def __init__(
        self,
        data: Any,
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        block_size: int = 32,
        scale_exp_bits: Optional[int] = None,
        scale_sig_bits: Optional[int] = None,
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
        self._impl = backend_module.MXTensor_(
            data,
            format=format,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits
        )
    
    def dequantize(self) -> Any:
        """Dequantize to original data type."""
        return self._impl.dequantize()
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        return self._impl.statistics()
    
    def __repr__(self):
        return f"MXTensor(backend={self.backend}, impl={self._impl})"


def compare_mx_formats(
    data: Any,
    formats: Optional[list] = None,
    block_size: int = 32
) -> None:
    """Compare different MX formats on same data."""
    backend = _resolve_backend(data)
    backend_module = _get_backend_module(backend)
    backend_module.compare_mx_formats(data, formats=formats, block_size=block_size)


def print_mx_format_table():
    """Print table of predefined MX formats."""
    print("="*100)
    print("OCP Microscaling (MX) Formats")
    print("="*100)
    
    header = (f"{'Name':<15} {'Element':<12} {'Block':<8} "
              f"{'Scale':<10} {'Compress FP16':<15} {'Total Bits':<12}")
    print(header)
    print("-"*100)
    
    for name, spec in MX_FORMATS.items():
        element_format = f"E{spec.exp_bits}M{spec.sig_bits}"
        row = (f"{spec.name:<15} "
               f"{element_format:<12} "
               f"{spec.block_size:<8} "
               f"{spec.scale_exp_bits}b{'':>6} "
               f"{spec.compression_vs_fp16:.2f}x{'':>11} "
               f"{spec.total_bits_per_block}")
        print(row)
    
    print("="*100)


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'MXSpec',
    'MXTensor',
    'MX_FORMATS',
    'create_mx_spec',
    'mx_quantize',
    'compare_mx_formats',
    'print_mx_format_table',
]