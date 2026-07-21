"""Backend-agnostic entry point for OCP microscaling (MX) formats.

MX quantization groups values into fixed-size blocks, stores one unsigned
power-of-two scale per block, and quantizes each scaled element to one of the
predefined OCP MX element formats. The built-in registry includes
``mxfp8_e5m2``, ``mxfp8_e4m3``, ``mxfp6_e3m2``, ``mxfp6_e2m3``,
``mxfp4_e2m1``, and ``mxint8``.

The front end supports NumPy, PyTorch, JAX, and TensorFlow. PyTorch, JAX, and
TensorFlow quantizer wrappers expose straight-through-estimator behavior for
training; NumPy is the reference implementation used for inference and tests.

References
----------
OCP Microscaling Formats (MX) v1.0 Specification.
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Examples
--------
>>> import numpy as np
>>> from pychop import mx_quantize
>>> x = np.random.randn(64)
>>> x_q = mx_quantize(x, format="mxfp8_e4m3")
"""

import os
from typing import Union, Tuple, Optional, Any, Dict
from dataclasses import dataclass


# ============================================================================
# Backend Detection (inline to avoid import issues)
# ============================================================================

def _detect_array_type(x: Any) -> str:
    """Detect the pychop backend implied by an input array.

    Parameters
    ----------
    x : Any
        Input array, tensor, or scalar.

    Returns
    -------
    backend : str
        One of ``"torch"``, ``"jax"``, ``"tensorflow"``, or ``"numpy"``.
    """
    module = type(x).__module__
    
    if "torch" in module:
        return "torch"
    if "jax" in module:
        return "jax"
    if "tensorflow" in module:
        return "tensorflow"
    return "numpy"


def _get_backend_env() -> str:
    """Return the backend name stored in ``chop_backend``."""
    return os.environ.get('chop_backend', 'auto')


# ============================================================================
# MX Format Specification (Backend-Independent)
# ============================================================================

@dataclass
class MXSpec:
    """
    Backend-independent OCP MX format specification.
    
    Attributes
    ----------
    name : str
        Format name, for example ``"MXFP8_E4M3"``.
    exp_bits : int
        Element exponent bits
    sig_bits : int
        Element significand bits excluding the implicit leading bit.
    block_size : int
        Number of elements sharing one scale.
    scale_exp_bits : int
        Scale exponent bits. OCP MX uses an E8M0 scale by default.
    scale_sig_bits : int
        Scale significand bits. This is normally zero for E8M0 scales.
    """
    name: str
    exp_bits: int
    sig_bits: int
    block_size: int = 32
    scale_exp_bits: int = 8
    scale_sig_bits: int = 0  # Scale is typically just exponent
    
    @property
    def element_bits(self) -> int:
        """Total stored bits per element."""
        return 1 + self.exp_bits + self.sig_bits
    
    @property
    def total_bits_per_block(self) -> int:
        """Total stored bits for one MX block, including the shared scale."""
        element_bits = self.element_bits * self.block_size
        scale_bits = self.scale_exp_bits + self.scale_sig_bits
        return element_bits + scale_bits
    
    @property
    def compression_vs_fp32(self) -> float:
        """Compression ratio relative to a dense FP32 block."""
        fp32_bits = 32 * self.block_size
        return fp32_bits / self.total_bits_per_block
    
    @property
    def compression_vs_fp16(self) -> float:
        """Compression ratio relative to a dense FP16 block."""
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
    """Create a custom MX format specification.

    Parameters
    ----------
    exp_bits : int
        Number of element exponent bits.
    sig_bits : int
        Number of stored element significand bits.
    block_size : int, default=32
        Number of elements sharing one block scale.
    scale_exp_bits : int, default=8
        Number of exponent bits in the shared scale.
    name : str, optional
        Custom format name. If omitted, a name is derived from the bit widths.

    Returns
    -------
    spec : MXSpec
        Backend-independent MX format specification.
    """
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
    """Resolve the backend requested for an MX operation.

    Parameters
    ----------
    X : Any, optional
        Input object used for auto detection when ``chop_backend="auto"``.

    Returns
    -------
    backend : str
        Resolved backend name.
    """
    env_backend = _get_backend_env()
    
    if env_backend == 'auto':
        if X is not None:
            return _detect_array_type(X)
        else:
            return 'numpy'
    
    if env_backend not in {'numpy', 'jax', 'torch', 'tensorflow'}:
        raise ValueError(
            f"Invalid backend: {env_backend}. "
            "Must be 'numpy', 'jax', 'torch', 'tensorflow', or 'auto'."
        )
    
    return env_backend


def _get_backend_module(backend: str):
    """Import the backend-specific MX implementation module.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    module
        Module implementing ``MXTensor_`` and ``mx_quantize`` for ``backend``.
    """
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
    elif backend == 'tensorflow':
        try:
            from .tf import mx_formats as backend_module
        except ImportError:
            raise ImportError(
                "TensorFlow backend not available. "
                "Install with: pip install tensorflow"
            )
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
    """Quantize an array or tensor to an OCP MX format.

    Parameters
    ----------
    data : array-like
        Input values. NumPy, PyTorch, JAX, and TensorFlow tensors are supported.
    format : str, MXSpec, or tuple of int, default="mxfp8_e4m3"
        Predefined format name, custom ``MXSpec``, or ``(exp_bits, sig_bits)``.
    block_size : int, default=32
        Number of elements sharing one scale.
    scale_exp_bits : int, optional
        Override for the shared scale exponent width.
    scale_sig_bits : int, optional
        Override for the shared scale significand width.
    backend : str, optional
        Backend override. If omitted, the active pychop backend or input type is
        used.

    Returns
    -------
    quantized : array-like
        Quantized-dequantized values in the same backend family as ``data``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(32)
    >>> x_q = mx_quantize(x, format="mxfp8_e4m3")
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
    """Backend-agnostic MX tensor wrapper.

    Parameters
    ----------
    data : array-like
        Input values.
    format : str, MXSpec, or tuple of int, default="mxfp8_e4m3"
        MX format specification.
    block_size : int, default=32
        Number of elements sharing one scale.
    scale_exp_bits : int, optional
        Override for scale exponent bits.
    scale_sig_bits : int, optional
        Override for scale significand bits.
    backend : str, optional
        Backend override. If omitted, backend auto detection is used.
    """
    
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
        """Return quantized-dequantized values in the backend-native type."""
        return self._impl.dequantize()
    
    def statistics(self) -> dict:
        """Return block count, compression, and range statistics."""
        return self._impl.statistics()
    
    def __repr__(self):
        return f"MXTensor(backend={self.backend}, impl={self._impl})"


def compare_mx_formats(
    data: Any,
    formats: Optional[list] = None,
    block_size: int = 32
) -> None:
    """Print a comparison of MX formats on the same data.

    Parameters
    ----------
    data : array-like
        Input values.
    formats : list of str, optional
        Format names to compare. If omitted, backend defaults are used.
    block_size : int, default=32
        Block size used for comparison.
    """
    backend = _resolve_backend(data)
    backend_module = _get_backend_module(backend)
    backend_module.compare_mx_formats(data, formats=formats, block_size=block_size)


def print_mx_format_table():
    """Print the predefined OCP MX format table."""
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




__all__ = [
    'MXSpec',
    'MXTensor',
    'MX_FORMATS',
    'create_mx_spec',
    'mx_quantize',
    'compare_mx_formats',
    'print_mx_format_table',
]
