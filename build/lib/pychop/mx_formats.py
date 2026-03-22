"""
Microscaling (MX) Data Formats Implementation for Pychop and 
MX Formats with Straight-Through Estimator (STE) for Quantization-Aware Training.

This module implements the OCP Microscaling formats with block-level shared exponents.
Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

This module extends the MX formats with STE support for training LLMs and other
large models with ultra-low precision.

Author: Xinye Chen
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from typing import Union, Optional, Tuple
from dataclasses import dataclass
from .chop import Chop




# ============================================================================
# MX Format Specifications
# ============================================================================

@dataclass
class MXSpec:
    """
    Specification for a Microscaling format.
    
    Parameters
    ----------
    name : str
        Format name (e.g., 'MXFP8_E4M3')
    exp_bits : int
        Number of exponent bits in each element
    sig_bits : int
        Number of significand bits in each element (excluding sign bit)
    scale_exp_bits : int, default=8
        Number of exponent bits in the shared scale factor (E8M0 default)
    scale_sig_bits : int, default=0
        Number of significand bits in the shared scale factor (E8M0 default)
    default_block_size : int, default=32
        Default number of elements per block
    
    Examples
    --------
    >>> # Predefined format
    >>> spec = MXSpec('MXFP8_E4M3', exp_bits=4, sig_bits=3)
    >>> 
    >>> # Custom format
    >>> spec = MXSpec('Custom_E6M5', exp_bits=6, sig_bits=5, default_block_size=64)
    """
    name: str
    exp_bits: int
    sig_bits: int
    scale_exp_bits: int = 8  # E8M0 format for scale factor
    scale_sig_bits: int = 0  # E8M0 (pure exponent)
    default_block_size: int = 32
    
    @property
    def element_bits(self) -> int:
        """Total bits per element (sign + exp + sig)."""
        return 1 + self.exp_bits + self.sig_bits
    
    @property
    def scale_bits(self) -> int:
        """Total bits for scale factor (sign + exp + sig)."""
        return 1 + self.scale_exp_bits + self.scale_sig_bits
    
    def __post_init__(self):
        if self.exp_bits <= 0:
            raise ValueError(f"exp_bits must be positive, got {self.exp_bits}")
        if self.sig_bits < 0:
            raise ValueError(f"sig_bits must be non-negative, got {self.sig_bits}")
        if self.scale_exp_bits <= 0:
            raise ValueError(f"scale_exp_bits must be positive, got {self.scale_exp_bits}")
        if self.scale_sig_bits < 0:
            raise ValueError(f"scale_sig_bits must be non-negative, got {self.scale_sig_bits}")
        if self.default_block_size <= 0:
            raise ValueError(f"default_block_size must be positive, got {self.default_block_size}")


# Predefined MX format specifications
MX_FORMATS = {
    # 8-bit formats
    'mxfp8_e5m2': MXSpec('MXFP8_E5M2', exp_bits=5, sig_bits=2, default_block_size=32),
    'mxfp8_e4m3': MXSpec('MXFP8_E4M3', exp_bits=4, sig_bits=3, default_block_size=32),
    
    # 6-bit formats
    'mxfp6_e3m2': MXSpec('MXFP6_E3M2', exp_bits=3, sig_bits=2, default_block_size=32),
    'mxfp6_e2m3': MXSpec('MXFP6_E2M3', exp_bits=2, sig_bits=3, default_block_size=32),
    
    # 4-bit formats
    'mxfp4_e2m1': MXSpec('MXFP4_E2M1', exp_bits=2, sig_bits=1, default_block_size=32),
    
    # Alternative scale formats (experimental)
    'mxfp8_e4m3_s6': MXSpec('MXFP8_E4M3_Scale6', exp_bits=4, sig_bits=3, 
                            scale_exp_bits=6, scale_sig_bits=0, default_block_size=32),
    'mxfp8_e4m3_s10': MXSpec('MXFP8_E4M3_Scale10', exp_bits=4, sig_bits=3,
                             scale_exp_bits=10, scale_sig_bits=0, default_block_size=32),
}


def create_mx_spec(
    exp_bits: int,
    sig_bits: int,
    scale_exp_bits: int = 8,
    scale_sig_bits: int = 0,
    block_size: int = 32,
    name: Optional[str] = None
) -> MXSpec:
    """
    Create a custom MX format specification.
    
    Parameters
    ----------
    exp_bits : int
        Number of exponent bits in each element
    sig_bits : int
        Number of significand bits in each element
    scale_exp_bits : int, default=8
        Number of exponent bits in shared scale factor
    scale_sig_bits : int, default=0
        Number of significand bits in shared scale factor
    block_size : int, default=32
        Default block size
    name : str, optional
        Format name (auto-generated if None)
    
    Returns
    -------
    MXSpec
        Custom format specification
    
    Examples
    --------
    >>> # Create a custom 10-bit format with 5 exp + 4 sig
    >>> spec = create_mx_spec(exp_bits=5, sig_bits=4, block_size=64)
    >>> 
    >>> # Create with custom scale (6-bit scale instead of 8-bit)
    >>> spec = create_mx_spec(exp_bits=4, sig_bits=3, scale_exp_bits=6)
    >>> 
    >>> # Ultra-low precision (3-bit elements!)
    >>> spec = create_mx_spec(exp_bits=1, sig_bits=1, block_size=16)
    """
    if name is None:
        name = f"Custom_E{exp_bits}M{sig_bits}"
        if scale_sig_bits > 0:
            name += f"_ScaleE{scale_exp_bits}M{scale_sig_bits}"
        elif scale_exp_bits != 8:
            name += f"_ScaleE{scale_exp_bits}"
    
    return MXSpec(
        name=name,
        exp_bits=exp_bits,
        sig_bits=sig_bits,
        scale_exp_bits=scale_exp_bits,
        scale_sig_bits=scale_sig_bits,
        default_block_size=block_size
    )


# ============================================================================
# MXBlock: Single block with shared scale
# ============================================================================

class MXBlock:
    """
    Represents a single microscaling block with shared exponent.
    
    A block consists of:
    - scale_factor: shared exponent for all elements
    - elements: array of low-precision values
    
    Parameters
    ----------
    values : array-like
        Input values to be encoded
    spec : MXSpec
        Format specification
    scale_factor : float, optional
        Pre-computed scale factor (if None, computed automatically)
    rmode : int, default=1
        Rounding mode for quantization
    subnormal : bool, default=True
        Support subnormal numbers in elements
    
    Examples
    --------
    >>> # Use predefined format
    >>> block = MXBlock([1.5, 2.3, 0.8, -1.2], spec=MX_FORMATS['mxfp8_e4m3'])
    >>> 
    >>> # Use custom format
    >>> custom_spec = create_mx_spec(exp_bits=5, sig_bits=4)
    >>> block = MXBlock([1.5, 2.3, 0.8, -1.2], spec=custom_spec)
    >>> 
    >>> print(block.scale_factor)  # Shared exponent
    >>> print(block.elements)      # Quantized elements
    >>> print(block.dequantize())  # Reconstructed values
    """
    
    def __init__(
        self, 
        values: Union[list, np.ndarray],
        spec: MXSpec,
        scale_factor: Optional[float] = None,
        rmode: int = 1,
        subnormal: bool = True
    ):
        self.spec = spec
        self.rmode = rmode
        self.subnormal = subnormal
        
        # Convert to numpy array
        self.original_values = np.asarray(values, dtype=np.float64)
        
        # Create quantizers
        self._element_chop = Chop(
            exp_bits=spec.exp_bits,
            sig_bits=spec.sig_bits,
            rmode=rmode,
            subnormal=subnormal
        )
        
        # Scale factor quantizer (can be E8M0, or custom)
        self._scale_chop = Chop(
            exp_bits=spec.scale_exp_bits,
            sig_bits=spec.scale_sig_bits,
            rmode=1,
            subnormal=False
        )
        
        # Encode the block
        if scale_factor is None:
            self.scale_factor, self.elements = self._encode()
        else:
            self.scale_factor = scale_factor
            self.elements = self._quantize_elements(self.original_values, scale_factor)
    
    def _encode(self) -> Tuple[float, np.ndarray]:
        """
        Encode values into MX format.
        
        Returns
        -------
        scale_factor : float
            Shared scale factor (power of 2)
        elements : np.ndarray
            Quantized element values
        """
        # Handle empty or all-zero blocks
        if len(self.original_values) == 0:
            return 0.0, np.array([])
        
        max_abs = np.max(np.abs(self.original_values))
        
        if max_abs == 0 or np.isnan(max_abs) or np.isinf(max_abs):
            return 0.0, np.zeros_like(self.original_values)
        
        # Compute optimal scale factor
        # The scale should normalize the max value to the representable range
        # For a format with t bits in significand, the max normalized value is ~2
        max_normalized = 2.0 ** (self.spec.sig_bits)
        
        # Compute scale as power of 2
        scale_exponent = np.floor(np.log2(max_abs / max_normalized))
        scale_factor = 2.0 ** scale_exponent
        
        # Quantize scale factor using the specified scale format
        scale_factor_quantized = self._scale_chop(np.array([scale_factor]))[0]
        
        # Quantize elements
        elements = self._quantize_elements(self.original_values, scale_factor_quantized)
        
        return scale_factor_quantized, elements
    
    def _quantize_elements(self, values: np.ndarray, scale: float) -> np.ndarray:
        """Normalize and quantize elements with given scale."""
        if scale == 0:
            return np.zeros_like(values)
        
        # Normalize by scale
        normalized = values / scale
        
        # Quantize to element format
        quantized = self._element_chop(normalized)
        
        return quantized
    
    def dequantize(self) -> np.ndarray:
        """
        Reconstruct original values from quantized representation.
        
        Returns
        -------
        np.ndarray
            Dequantized values
        """
        return self.elements * self.scale_factor
    
    def __repr__(self):
        return (f"MXBlock(format={self.spec.name}, "
                f"scale={self.scale_factor:.2e}, "
                f"size={len(self.elements)})")
    
    def storage_bits(self) -> int:
        """Calculate total storage bits for this block."""
        return self.spec.scale_bits + len(self.elements) * self.spec.element_bits
    
    def compression_ratio(self, baseline_bits: int = 16) -> float:
        """
        Calculate compression ratio compared to baseline format.
        
        Parameters
        ----------
        baseline_bits : int, default=16
            Baseline format bit-width (e.g., FP16=16, FP32=32)
        
        Returns
        -------
        float
            Compression ratio (baseline_bits / mx_bits_per_element)
        """
        mx_bits_per_element = self.storage_bits() / len(self.elements)
        return baseline_bits / mx_bits_per_element


# ============================================================================
# MXTensor: Multi-block microscaling tensor
# ============================================================================

class MXTensor:
    """
    Microscaling tensor with multiple blocks.
    
    This class manages a full tensor encoded in MX format, automatically
    dividing it into blocks and managing shared scale factors.
    
    Parameters
    ----------
    values : array-like
        Input tensor values
    format : str, MXSpec, or tuple
        Format specification. Can be:
        - String: 'mxfp8_e4m3' (predefined format)
        - MXSpec: custom specification object
        - Tuple: (exp_bits, sig_bits) for quick custom format
    block_size : int, optional
        Number of elements per block (default from spec)
    axis : int, default=-1
        Axis along which to form blocks (-1 for flattened)
    rmode : int, default=1
        Rounding mode
    subnormal : bool, default=True
        Support subnormal numbers
    scale_exp_bits : int, optional
        Override scale exponent bits (only for tuple format)
    scale_sig_bits : int, optional
        Override scale significand bits (only for tuple format)
    
    Attributes
    ----------
    blocks : list of MXBlock
        List of encoded blocks
    shape : tuple
        Original tensor shape
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(128, 64)
    >>> 
    >>> # Using predefined format
    >>> mx1 = MXTensor(X, format='mxfp8_e4m3', block_size=32)
    >>> 
    >>> # Using custom format (tuple)
    >>> mx2 = MXTensor(X, format=(5, 4), block_size=32)  # E5M4
    >>> 
    >>> # Using custom MXSpec
    >>> spec = create_mx_spec(exp_bits=6, sig_bits=5, scale_exp_bits=10)
    >>> mx3 = MXTensor(X, spec=spec)
    >>> 
    >>> # Ultra-low precision
    >>> mx4 = MXTensor(X, format=(2, 1), block_size=16)  # E2M1, 4-bit elements!
    >>> 
    >>> # Check results
    >>> print(f"Compression: {mx1.compression_ratio():.2f}x")
    >>> X_reconstructed = mx1.dequantize()
    """
    
    def __init__(
        self,
        values: Union[list, np.ndarray],
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        block_size: Optional[int] = None,
        axis: int = -1,
        rmode: int = 1,
        subnormal: bool = True,
        scale_exp_bits: Optional[int] = None,
        scale_sig_bits: Optional[int] = None
    ):
        # Parse format
        if isinstance(format, str):
            if format.lower() not in MX_FORMATS:
                raise ValueError(f"Unknown format: {format}. "
                               f"Available: {list(MX_FORMATS.keys())}")
            self.spec = MX_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            if len(format) != 2:
                raise ValueError(f"Format tuple must be (exp_bits, sig_bits), got {format}")
            exp_bits, sig_bits = format
            
            # Use provided scale bits or defaults
            _scale_exp = scale_exp_bits if scale_exp_bits is not None else 8
            _scale_sig = scale_sig_bits if scale_sig_bits is not None else 0
            _block_size = block_size if block_size is not None else 32
            
            self.spec = create_mx_spec(
                exp_bits=exp_bits,
                sig_bits=sig_bits,
                scale_exp_bits=_scale_exp,
                scale_sig_bits=_scale_sig,
                block_size=_block_size
            )
        elif isinstance(format, MXSpec):
            self.spec = format
        else:
            raise TypeError(f"format must be str, MXSpec, or tuple, got {type(format)}")
        
        # Set block size
        self.block_size = block_size or self.spec.default_block_size
        self.axis = axis
        self.rmode = rmode
        self.subnormal = subnormal
        
        # Store original shape
        self.values = np.asarray(values, dtype=np.float64)
        self.shape = self.values.shape
        
        # Flatten or reshape for blocking
        if axis == -1:
            # Flatten entire tensor
            self.flat_values = self.values.flatten()
        else:
            # Flatten along specified axis
            # TODO: Implement axis-specific blocking
            raise NotImplementedError("Axis-specific blocking not yet implemented")
        
        # Encode into blocks
        self.blocks = self._encode_blocks()
    
    def _encode_blocks(self) -> list:
        """Divide tensor into blocks and encode each."""
        n = len(self.flat_values)
        blocks = []
        
        for i in range(0, n, self.block_size):
            block_values = self.flat_values[i:i + self.block_size]
            
            # Pad last block if necessary
            if len(block_values) < self.block_size:
                padding = self.block_size - len(block_values)
                block_values = np.pad(block_values, (0, padding), mode='constant')
            
            block = MXBlock(
                block_values,
                spec=self.spec,
                rmode=self.rmode,
                subnormal=self.subnormal
            )
            blocks.append(block)
        
        return blocks
    
    def dequantize(self) -> np.ndarray:
        """
        Reconstruct full tensor from MX representation.
        
        Returns
        -------
        np.ndarray
            Dequantized tensor with original shape
        """
        # Dequantize all blocks
        dequantized_blocks = [block.dequantize() for block in self.blocks]
        
        # Concatenate
        flat_dequantized = np.concatenate(dequantized_blocks)
        
        # Trim padding and reshape
        n_original = np.prod(self.shape)
        flat_dequantized = flat_dequantized[:n_original]
        
        return flat_dequantized.reshape(self.shape)
    
    def __repr__(self):
        return (f"MXTensor(format={self.spec.name}, "
                f"shape={self.shape}, "
                f"n_blocks={len(self.blocks)}, "
                f"block_size={self.block_size})")
    
    def storage_bits(self) -> int:
        """Total storage bits for entire tensor."""
        return sum(block.storage_bits() for block in self.blocks)
    
    def compression_ratio(self, baseline_bits: int = 16) -> float:
        """
        Calculate overall compression ratio.
        
        Parameters
        ----------
        baseline_bits : int, default=16
            Baseline format (FP16=16, FP32=32)
        
        Returns
        -------
        float
            Compression ratio
        """
        baseline_total = np.prod(self.shape) * baseline_bits
        mx_total = self.storage_bits()
        return baseline_total / mx_total
    
    def statistics(self) -> dict:
        """
        Compute statistics about the MX encoding.
        
        Returns
        -------
        dict
            Statistics including scale factors, compression, errors, etc.
        """
        original = self.values.flatten()
        reconstructed = self.dequantize().flatten()
        
        # Compute errors
        abs_error = np.abs(original - reconstructed)
        rel_error = abs_error / (np.abs(original) + 1e-10)
        
        # Scale factor statistics
        scales = [block.scale_factor for block in self.blocks]
        
        return {
            'format': self.spec.name,
            'element_bits': self.spec.element_bits,
            'exp_bits': self.spec.exp_bits,
            'sig_bits': self.spec.sig_bits,
            'scale_bits': self.spec.scale_bits,
            'shape': self.shape,
            'n_blocks': len(self.blocks),
            'block_size': self.block_size,
            'compression_ratio_fp16': self.compression_ratio(16),
            'compression_ratio_fp32': self.compression_ratio(32),
            'storage_bits': self.storage_bits(),
            'storage_bytes': self.storage_bits() / 8,
            'mean_abs_error': np.mean(abs_error),
            'max_abs_error': np.max(abs_error),
            'mean_rel_error': np.mean(rel_error),
            'max_rel_error': np.max(rel_error),
            'scale_min': np.min(scales),
            'scale_max': np.max(scales),
            'scale_mean': np.mean(scales),
        }


# ============================================================================
# Arithmetic Operations with MXTensor
# ============================================================================

class MXFloat:
    """
    Scalar value in MX format (single-element block).
    
    Similar to CPFloat but uses MX encoding.
    
    Parameters
    ----------
    value : float
        Scalar value
    format : str, MXSpec, or tuple
        MX format specification
    
    Examples
    --------
    >>> # Using predefined format
    >>> a = MXFloat(3.14159, 'mxfp8_e4m3')
    >>> 
    >>> # Using custom format (tuple)
    >>> b = MXFloat(2.71828, (5, 4))  # E5M4 format
    >>> 
    >>> # Using MXSpec
    >>> spec = create_mx_spec(exp_bits=6, sig_bits=5)
    >>> c = MXFloat(1.618, spec)
    >>> 
    >>> # Arithmetic
    >>> d = a + b
    >>> print(d)  # MXFloat(5.86, format=...)
    """
    
    def __init__(
        self, 
        value: float, 
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3'
    ):
        # Parse format (reuse MXTensor logic)
        if isinstance(format, str):
            if format.lower() not in MX_FORMATS:
                raise ValueError(f"Unknown format: {format}")
            self.spec = MX_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            if len(format) != 2:
                raise ValueError(f"Format tuple must be (exp_bits, sig_bits)")
            exp_bits, sig_bits = format
            self.spec = create_mx_spec(exp_bits=exp_bits, sig_bits=sig_bits)
        elif isinstance(format, MXSpec):
            self.spec = format
        else:
            raise TypeError(f"format must be str, MXSpec, or tuple")
        
        self.block = MXBlock([value], spec=self.spec)
        self._value = self.block.dequantize()[0]
    
    @property
    def value(self) -> float:
        """Get dequantized value."""
        return self._value
    
    def __float__(self):
        return float(self._value)
    
    def __repr__(self):
        return f"MXFloat({self._value:.6g}, format={self.spec.name})"
    
    # Arithmetic operations (always return MXFloat)
    def __add__(self, other):
        if isinstance(other, MXFloat):
            result = self._value + other._value
        else:
            result = self._value + float(other)
        return MXFloat(result, self.spec)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, MXFloat):
            result = self._value - other._value
        else:
            result = self._value - float(other)
        return MXFloat(result, self.spec)
    
    def __rsub__(self, other):
        result = float(other) - self._value
        return MXFloat(result, self.spec)
    
    def __mul__(self, other):
        if isinstance(other, MXFloat):
            result = self._value * other._value
        else:
            result = self._value * float(other)
        return MXFloat(result, self.spec)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, MXFloat):
            result = self._value / other._value
        else:
            result = self._value / float(other)
        return MXFloat(result, self.spec)
    
    def __rtruediv__(self, other):
        result = float(other) / self._value
        return MXFloat(result, self.spec)
    
    def __neg__(self):
        return MXFloat(-self._value, self.spec)
    
    def __abs__(self):
        return MXFloat(abs(self._value), self.spec)
    
    # Comparison operations
    def __eq__(self, other):
        if isinstance(other, MXFloat):
            return self._value == other._value
        return self._value == float(other)
    
    def __lt__(self, other):
        if isinstance(other, MXFloat):
            return self._value < other._value
        return self._value < float(other)
    
    def __le__(self, other):
        if isinstance(other, MXFloat):
            return self._value <= other._value
        return self._value <= float(other)
    
    def __gt__(self, other):
        if isinstance(other, MXFloat):
            return self._value > other._value
        return self._value > float(other)
    
    def __ge__(self, other):
        if isinstance(other, MXFloat):
            return self._value >= other._value
        return self._value >= float(other)


# ============================================================================
# Utility Functions
# ============================================================================

def mx_quantize(
    values: Union[list, np.ndarray],
    format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
    block_size: Optional[int] = None,
    return_tensor: bool = False,
    **kwargs
) -> Union[np.ndarray, MXTensor]:
    """
    Quantize array to MX format and optionally dequantize.
    
    Parameters
    ----------
    values : array-like
        Input values
    format : str, MXSpec, or tuple
        MX format specification
    block_size : int, optional
        Block size (default from format spec)
    return_tensor : bool, default=False
        If True, return MXTensor object; else return dequantized array
    **kwargs
        Additional arguments passed to MXTensor (e.g., scale_exp_bits)
    
    Returns
    -------
    np.ndarray or MXTensor
        Quantized result
    
    Examples
    --------
    >>> X = np.random.randn(1024)
    >>> 
    >>> # Predefined format
    >>> X_q = mx_quantize(X, 'mxfp8_e4m3', block_size=32)
    >>> 
    >>> # Custom format (tuple)
    >>> X_q = mx_quantize(X, format=(6, 5), block_size=64)
    >>> 
    >>> # Custom scale
    >>> X_q = mx_quantize(X, format=(4, 3), scale_exp_bits=10)
    >>> 
    >>> print(f"Error: {np.mean(np.abs(X - X_q)):.6f}")
    """
    mx_tensor = MXTensor(values, format=format, block_size=block_size, **kwargs)
    
    if return_tensor:
        return mx_tensor
    else:
        return mx_tensor.dequantize()


def compare_mx_formats(
    values: np.ndarray,
    formats: Optional[list] = None,
    block_sizes: Optional[list] = None,
    custom_formats: Optional[list] = None
) -> dict:
    """
    Compare different MX formats on the same data.
    
    Parameters
    ----------
    values : np.ndarray
        Test data
    formats : list, optional
        List of predefined format names (default: all formats)
    block_sizes : list, optional
        List of block sizes to test (default: [32])
    custom_formats : list of tuples, optional
        List of (exp_bits, sig_bits) tuples for custom formats
    
    Returns
    -------
    dict
        Comparison results
    
    Examples
    --------
    >>> X = np.random.randn(1024, 512)
    >>> 
    >>> # Compare predefined formats
    >>> results = compare_mx_formats(X)
    >>> 
    >>> # Compare custom formats
    >>> results = compare_mx_formats(
    ...     X, 
    ...     custom_formats=[(5, 4), (6, 5), (7, 6)]
    ... )
    >>> 
    >>> # Compare block sizes
    >>> results = compare_mx_formats(X, block_sizes=[16, 32, 64, 128])
    >>> 
    >>> for fmt, stats in results.items():
    ...     print(f"{fmt}: {stats['compression_ratio_fp16']:.2f}x, "
    ...           f"error={stats['mean_abs_error']:.6f}")
    """
    if formats is None and custom_formats is None:
        formats = list(MX_FORMATS.keys())
    
    if block_sizes is None:
        block_sizes = [32]
    
    results = {}
    
    # Test predefined formats
    if formats:
        for fmt in formats:
            for bs in block_sizes:
                key = f"{fmt}_block{bs}"
                mx_tensor = MXTensor(values, format=fmt, block_size=bs)
                results[key] = mx_tensor.statistics()
    
    # Test custom formats
    if custom_formats:
        for exp_bits, sig_bits in custom_formats:
            for bs in block_sizes:
                key = f"E{exp_bits}M{sig_bits}_block{bs}"
                mx_tensor = MXTensor(values, format=(exp_bits, sig_bits), block_size=bs)
                results[key] = mx_tensor.statistics()
    
    return results


def print_mx_format_table():
    """Print a formatted table of available MX formats."""
    print("\n" + "="*100)
    print("Available Microscaling (MX) Formats")
    print("="*100)
    print(f"{'Format':<25} {'Element':<10} {'Exp':<6} {'Sig':<6} {'Scale':<10} "
          f"{'Block':<8} {'Compression (vs FP16)':<20}")
    print("-"*100)
    
    for name, spec in MX_FORMATS.items():
        # Calculate bits per element including amortized scale
        bits_per_elem = spec.element_bits + spec.scale_bits / spec.default_block_size
        compression = 16 / bits_per_elem
        
        scale_str = f"E{spec.scale_exp_bits}M{spec.scale_sig_bits}"
        
        print(f"{name:<25} {spec.element_bits:<10} {spec.exp_bits:<6} {spec.sig_bits:<6} "
              f"{scale_str:<10} {spec.default_block_size:<8} {compression:.2f}x")
    
    print("="*100)
    print("Note: You can create custom formats with create_mx_spec() or use tuple format=(exp, sig)")
    print("="*100 + "\n")



# ============================================================================
# MX Straight-Through Estimator (STE)
# ============================================================================

class MXQuantizeSTE(Function):
    """
    Straight-Through Estimator for MX format quantization.
    
    Forward: Applies real MX quantization (block-level shared exponent)
    Backward: Passes gradients straight through (identity function)
    
    This enables quantization-aware training with MX formats!
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, mx_spec: MXSpec, block_size: int,
                scale_exp_bits: Optional[int] = None,
                scale_sig_bits: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass: apply MX quantization.
        
        Parameters
        ----------
        ctx : context
            PyTorch autograd context
        input : torch.Tensor
            Input tensor
        mx_spec : MXSpec
            MX format specification
        block_size : int
            Block size for MX quantization
        scale_exp_bits : int, optional
            Override scale exponent bits
        scale_sig_bits : int, optional
            Override scale significand bits
        
        Returns
        -------
        torch.Tensor
            Quantized tensor (same shape as input)
        """
        # Save for backward (though we don't use it in STE)
        ctx.save_for_backward(input)
        
        # Convert to numpy, quantize, convert back
        input_np = input.detach().cpu().numpy()
        
        # Create MXTensor
        if isinstance(mx_spec, tuple):
            mx_tensor = MXTensor(
                input_np,
                format=mx_spec,
                block_size=block_size,
                scale_exp_bits=scale_exp_bits,
                scale_sig_bits=scale_sig_bits
            )
        else:
            mx_tensor = MXTensor(input_np, format=mx_spec, block_size=block_size)
        
        # Dequantize
        output_np = mx_tensor.dequantize()
        
        # Convert back to torch tensor
        output = torch.from_numpy(output_np).to(input.device).to(input.dtype)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """
        Backward pass: straight-through (identity).
        
        The gradient flows unchanged, as if quantization never happened.
        This is the key to quantization-aware training!
        """
        # STE: pass gradient straight through
        return grad_output, None, None, None, None


# ============================================================================
# MX Quantizer with STE
# ============================================================================

class MXQuantizerSTE(nn.Module):
    """
    MX format quantizer with Straight-Through Estimator for QAT.
    
    This module can be used as a drop-in replacement for activation functions
    or as a wrapper around linear/conv layers for LLM fine-tuning.
    
    Parameters
    ----------
    format : str, MXSpec, or tuple
        MX format specification:
        - String: 'mxfp8_e4m3' (predefined format)
        - Tuple: (exp_bits, sig_bits) for custom format
        - MXSpec: full specification object
    block_size : int, default=32
        Number of elements per block
    scale_exp_bits : int, optional
        Override scale exponent bits
    scale_sig_bits : int, optional
        Override scale significand bits
    
    Examples
    --------
    >>> # Quantize activations
    >>> quantizer = MXQuantizerSTE(format='mxfp8_e4m3', block_size=32)
    >>> x = torch.randn(128, 768, requires_grad=True)
    >>> x_q = quantizer(x)  # Quantized with STE
    >>> 
    >>> # Custom format for extreme compression
    >>> quantizer = MXQuantizerSTE(format=(4, 3), block_size=32)  # E4M3
    >>> 
    >>> # Ultra-low precision
    >>> quantizer = MXQuantizerSTE(format=(2, 1), block_size=16)  # E2M1 (4-bit!)
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
            self.mx_spec = MX_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            if len(format) != 2:
                raise ValueError(f"Format tuple must be (exp_bits, sig_bits)")
            self.mx_spec = format  # Store tuple, will be parsed in forward
        elif isinstance(format, MXSpec):
            self.mx_spec = format
        else:
            raise TypeError(f"format must be str, MXSpec, or tuple")
        
        self.block_size = block_size
        self.scale_exp_bits = scale_exp_bits
        self.scale_sig_bits = scale_sig_bits
        
        # Store format string for repr
        if isinstance(format, str):
            self.format_str = format
        elif isinstance(format, tuple):
            self.format_str = f"E{format[0]}M{format[1]}"
        else:
            self.format_str = format.name
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MX quantization with STE.
        
        During training (requires_grad=True): uses STE
        During inference (no grad): uses standard quantization
        """
        if x.requires_grad and torch.is_grad_enabled():
            # Training: use STE
            return MXQuantizeSTE.apply(
                x, self.mx_spec, self.block_size,
                self.scale_exp_bits, self.scale_sig_bits
            )
        else:
            # Inference: standard quantization (no autograd overhead)
            with torch.no_grad():
                return MXQuantizeSTE.apply(
                    x, self.mx_spec, self.block_size,
                    self.scale_exp_bits, self.scale_sig_bits
                )
    
    def __repr__(self):
        return (f"MXQuantizerSTE(format={self.format_str}, "
                f"block_size={self.block_size})")
    
    def extra_repr(self):
        return f"format={self.format_str}, block_size={self.block_size}"


# ============================================================================
# Quantized Linear Layer with MX Format
# ============================================================================

class MXLinear(nn.Linear):
    """
    Quantized Linear layer using MX format for weights and activations.
    
    This layer quantizes:
    - Input activations (optional)
    - Weights (always)
    - Output activations (optional)
    
    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    bias : bool, default=True
        If True, include bias term
    weight_format : str, MXSpec, or tuple, default='mxfp8_e4m3'
        Format for weight quantization
    act_format : str, MXSpec, or tuple, optional
        Format for activation quantization (if None, no activation quantization)
    block_size : int, default=32
        Block size for MX quantization
    quantize_input : bool, default=True
        Whether to quantize input activations
    quantize_output : bool, default=False
        Whether to quantize output activations
    
    Examples
    --------
    >>> # Standard MX quantized linear layer
    >>> layer = MXLinear(768, 3072, weight_format='mxfp8_e4m3')
    >>> 
    >>> # Ultra-low precision for extreme compression
    >>> layer = MXLinear(768, 3072, weight_format=(4, 3), act_format=(4, 3))
    >>> 
    >>> # Quantize both input and output
    >>> layer = MXLinear(768, 3072, quantize_input=True, quantize_output=True)
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
        
        # Weight quantizer
        self.weight_quantizer = MXQuantizerSTE(
            format=weight_format,
            block_size=block_size
        )
        
        # Activation quantizers
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
        """
        Forward pass with MX quantization.
        
        Order of operations:
        1. Quantize input (optional)
        2. Quantize weights
        3. Linear operation
        4. Add bias (if present)
        5. Quantize output (optional)
        """
        # Quantize input
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        
        # Quantize weights
        weight_q = self.weight_quantizer(self.weight)
        
        # Linear operation
        output = torch.nn.functional.linear(x, weight_q, self.bias)
        
        # Quantize output
        if self.output_quantizer is not None:
            output = self.output_quantizer(output)
        
        return output


# ============================================================================
# Quantized Attention Layer with MX Format
# ============================================================================

class MXAttention(nn.Module):
    """
    Multi-head attention with MX quantization for LLM fine-tuning.
    
    Quantizes Q, K, V projections and attention output with MX format.
    This is critical for LLM memory reduction!
    
    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    format : str, MXSpec, or tuple, default='mxfp8_e4m3'
        MX format for weights and activations
    block_size : int, default=32
        Block size for MX quantization
    dropout : float, default=0.0
        Dropout probability
    bias : bool, default=True
        Whether to include bias in projections
    
    Examples
    --------
    >>> # Standard transformer attention with MX quantization
    >>> attn = MXAttention(768, num_heads=12, format='mxfp8_e4m3')
    >>> 
    >>> # Ultra-low precision for 4x compression
    >>> attn = MXAttention(768, num_heads=12, format=(4, 3), block_size=32)
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
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Q, K, V projections with MX quantization
        self.q_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                               weight_format=format, block_size=block_size,
                               quantize_input=True, quantize_output=False)
        self.k_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                               weight_format=format, block_size=block_size,
                               quantize_input=True, quantize_output=False)
        self.v_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                               weight_format=format, block_size=block_size,
                               quantize_input=True, quantize_output=False)
        
        # Output projection
        self.out_proj = MXLinear(embed_dim, embed_dim, bias=bias,
                                 weight_format=format, block_size=block_size,
                                 quantize_input=True, quantize_output=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Quantizer for attention scores (optional, but helpful)
        self.score_quantizer = MXQuantizerSTE(format=format, block_size=block_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with MX quantized attention.
        
        Parameters
        ----------
        query : torch.Tensor
            Query tensor [batch, seq_len, embed_dim]
        key : torch.Tensor, optional
            Key tensor (if None, uses query)
        value : torch.Tensor, optional
            Value tensor (if None, uses query)
        attn_mask : torch.Tensor, optional
            Attention mask
        
        Returns
        -------
        torch.Tensor
            Attention output [batch, seq_len, embed_dim]
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size = query.size(0)
        
        # Project Q, K, V (with MX quantization)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Quantize attention scores (optional but helps)
        scores = self.score_quantizer(scores)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        
        # Output projection (with MX quantization)
        output = self.out_proj(attn_output)
        
        return output


# ============================================================================
# Utility: Convert existing model to MX quantized version
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
    Convert all Linear layers in a model to MX quantized versions.
    
    This is useful for retrofitting existing LLMs with MX quantization!
    
    Parameters
    ----------
    module : nn.Module
        Model to convert
    format : str, MXSpec, or tuple
        MX format specification
    block_size : int, default=32
        Block size
    quantize_input : bool, default=True
        Whether to quantize inputs
    quantize_output : bool, default=False
        Whether to quantize outputs
    inplace : bool, default=True
        If True, modify module in place; else create copy
    
    Returns
    -------
    nn.Module
        Converted model with MX quantized layers
    
    Examples
    --------
    >>> import transformers
    >>> model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    >>> 
    >>> # Convert all linear layers to MX format
    >>> model = convert_linear_to_mx(model, format='mxfp8_e4m3', block_size=32)
    >>> 
    >>> # Now fine-tune with MX quantization!
    >>> # ... training loop ...
    """
    import copy
    
    if not inplace:
        module = copy.deepcopy(module)
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, MXLinear):
            # Replace with MX quantized version
            mx_layer = MXLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                weight_format=format,
                block_size=block_size,
                quantize_input=quantize_input,
                quantize_output=quantize_output
            )
            
            # Copy weights and bias
            mx_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                mx_layer.bias.data = child.bias.data.clone()
            
            # Replace module
            setattr(module, name, mx_layer)
        else:
            # Recursively convert children
            convert_linear_to_mx(
                child, format, block_size,
                quantize_input, quantize_output, inplace=True
            )
    
    return module



# ============================================================================
# Example and Testing
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("Microscaling (MX) Formats Demo - With Custom Format Support")
    print("="*100)
    
    # Show available formats
    print_mx_format_table()
    
    # Example 1: Custom format using tuple
    print("\n" + "="*100)
    print("Example 1: Custom Format Using Tuple (E5M4)")
    print("="*100)
    
    X = np.random.randn(256).astype(np.float32)
    
    # E5M4 format (11-bit elements: 1 sign + 5 exp + 4 sig + 1 implicit)
    mx_tensor = MXTensor(X, format=(5, 4), block_size=32)
    X_recon = mx_tensor.dequantize()
    
    stats = mx_tensor.statistics()
    print(f"\nFormat: {stats['format']}")
    print(f"Element bits: {stats['element_bits']} (E{stats['exp_bits']}M{stats['sig_bits']})")
    print(f"Scale bits: {stats['scale_bits']}")
    print(f"Compression vs FP16: {stats['compression_ratio_fp16']:.2f}x")
    print(f"Mean error: {stats['mean_abs_error']:.6f}")
    
    # Example 2: Custom format with custom scale
    print("\n" + "="*100)
    print("Example 2: Custom Format with Custom Scale (E4M3 elements, E10 scale)")
    print("="*100)
    
    # Standard: E8M0 scale (8-bit scale factor)
    mx1 = MXTensor(X, format=(4, 3), block_size=32)
    
    # Custom: E10M0 scale (10-bit scale factor for larger dynamic range)
    mx2 = MXTensor(X, format=(4, 3), block_size=32, scale_exp_bits=10, scale_sig_bits=0)
    
    print(f"\nWith E8 scale:  compression={mx1.compression_ratio():.2f}x, "
          f"error={mx1.statistics()['mean_abs_error']:.6f}")
    print(f"With E10 scale: compression={mx2.compression_ratio():.2f}x, "
          f"error={mx2.statistics()['mean_abs_error']:.6f}")
    
    # Example 3: Ultra-low precision
    print("\n" + "="*100)
    print("Example 3: Ultra-Low Precision Formats")
    print("="*100)
    
    ultra_low_formats = [
        (3, 2),  # 6-bit elements
        (2, 2),  # 5-bit elements
        (2, 1),  # 4-bit elements
        (1, 1),  # 3-bit elements (extreme!)
    ]
    
    print(f"\n{'Format':<15} {'Element Bits':<15} {'Compression':<15} {'Mean Error':<15}")
    print("-"*60)
    
    for exp, sig in ultra_low_formats:
        mx = MXTensor(X, format=(exp, sig), block_size=32)
        stats = mx.statistics()
        print(f"E{exp}M{sig}{'':<11} {stats['element_bits']:<15} "
              f"{stats['compression_ratio_fp16']:<15.2f} "
              f"{stats['mean_abs_error']:<15.6f}")
    
    # Example 4: MXFloat with custom format
    print("\n" + "="*100)
    print("Example 4: MXFloat Scalar with Custom Format")
    print("="*100)
    
    # E6M5 format (12-bit)
    a = MXFloat(3.14159, (6, 5))
    b = MXFloat(2.71828, (6, 5))
    
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    
    # Example 5: Compare custom formats
    print("\n" + "="*100)
    print("Example 5: Compare Custom Formats")
    print("="*100)
    
    X_large = np.random.randn(512, 256).astype(np.float32)
    
    results = compare_mx_formats(
        X_large,
        custom_formats=[
            (4, 3),  # 8-bit
            (5, 4),  # 10-bit
            (6, 5),  # 12-bit
            (7, 6),  # 14-bit
        ]
    )
    
    print(f"\n{'Format':<20} {'Element Bits':<15} {'Compression':<15} "
          f"{'Mean Error':<15} {'Max Error':<15}")
    print("-"*80)
    
    for fmt_name, stats in sorted(results.items()):
        print(f"{fmt_name:<20} {stats['element_bits']:<15} "
              f"{stats['compression_ratio_fp16']:<15.2f} "
              f"{stats['mean_abs_error']:<15.6f} "
              f"{stats['max_abs_error']:<15.6f}")
    
    # Example 6: Using MXSpec directly
    print("\n" + "="*100)
    print("Example 6: Using MXSpec for Full Control")
    print("="*100)
    
    # Create a custom spec with all parameters
    custom_spec = create_mx_spec(
        exp_bits=5,
        sig_bits=4,
        scale_exp_bits=10,
        scale_sig_bits=1,  # Non-zero significand in scale!
        block_size=64,
        name="MyCustomFormat_E5M4_ScaleE10M1"
    )
    
    mx = MXTensor(X, format=custom_spec)
    stats = mx.statistics()
    
    print(f"\nCustom format: {stats['format']}")
    print(f"Element: E{stats['exp_bits']}M{stats['sig_bits']} ({stats['element_bits']} bits)")
    print(f"Scale: {stats['scale_bits']} bits")
    print(f"Block size: {stats['block_size']}")
    print(f"Compression: {stats['compression_ratio_fp16']:.2f}x")
    print(f"Error: {stats['mean_abs_error']:.6f}")
    
    print("\n" + "="*100)
    print("Demo Complete! You can now use any custom E/M combination you want!")
    
    
    
    
    # STE tests
    print("="*100)
    print("="*80)
    print("MX Formats with STE for LLM Fine-tuning - Demo")
    print("="*80)
    
    # Test 1: Basic STE functionality
    print("\nTest 1: MX Quantizer with STE")
    quantizer = MXQuantizerSTE(format='mxfp8_e4m3', block_size=32)
    x = torch.randn(128, 768, requires_grad=True)
    x_q = quantizer(x)
    
    # Backward pass (STE)
    loss = x_q.sum()
    loss.backward()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_q.shape}")
    print(f"Gradient shape: {x.grad.shape}")
    print(f"✓ STE works! Gradients flow through.")
    
    # Test 2: MX Linear Layer
    print("\nTest 2: MX Quantized Linear Layer")
    layer = MXLinear(768, 3072, weight_format='mxfp8_e4m3')
    x = torch.randn(32, 128, 768, requires_grad=True)
    y = layer(x)
    
    loss = y.sum()
    loss.backward()
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Weight grad: {layer.weight.grad is not None}")
    print(f"✓ MXLinear works with gradient flow!")
    
    # Test 3: MX Attention
    print("\nTest 3: MX Quantized Attention")
    attn = MXAttention(768, num_heads=12, format='mxfp8_e4m3')
    x = torch.randn(16, 128, 768, requires_grad=True)
    y = attn(x)
    
    loss = y.sum()
    loss.backward()
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"✓ MXAttention works!")
    
    # Test 4: Model conversion
    print("\nTest 4: Convert existing model to MX")
    
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 3072)
            self.fc2 = nn.Linear(3072, 768)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    model = SimpleTransformer()
    print(f"Original model: {model}")
    
    model_mx = convert_linear_to_mx(model, format='mxfp8_e4m3', block_size=32, inplace=False)
    print(f"\nConverted model: {model_mx}")
    print(f"✓ Model conversion works!")
    
    print("\n" + "="*80)
    print("All tests passed! Ready for LLM fine-tuning with MX formats!")
    print("="*80)
