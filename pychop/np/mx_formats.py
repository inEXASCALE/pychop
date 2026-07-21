"""NumPy reference backend for OCP MX quantization.

This module implements the OCP MX forward semantics used by all wrappers:
E8M0-style block scale selection, round-to-nearest ties-to-even element
quantization, finite saturation, FP subnormal support where defined, and
two's-complement scaled-integer behavior for ``mxint8``.
"""


import numpy as np
import math
from typing import Union, Tuple, Optional, List
from dataclasses import replace
import warnings

# Import shared spec
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mx_formats import MXSpec, MX_FORMATS, create_mx_spec


def _scale_exponent_bounds(scale_exp_bits: int) -> Tuple[int, int]:
    """Return the exponent range for an unsigned E<M>0 scale type."""
    bias = 2 ** (scale_exp_bits - 1) - 1
    return -bias, (2 ** scale_exp_bits - 2) - bias


def _mx_float_params(spec: MXSpec) -> dict:
    """Return OCP-style finite range parameters for an MX element format."""
    if spec.exp_bits == 0:
        int_scale = 2 ** (spec.element_bits - 2)
        return {
            "max": (2 ** (spec.element_bits - 1) - 1) / int_scale,
            "min": -(2 ** (spec.element_bits - 1)) / int_scale,
            "emax": 0,
        }

    bias = 2 ** (spec.exp_bits - 1) - 1
    max_exp_field = 2 ** spec.exp_bits - 1
    max_sig_field = 2 ** spec.sig_bits - 1
    name = spec.name.upper()

    if name == "MXFP8_E5M2":
        max_exp_field -= 1
    elif name == "MXFP8_E4M3":
        max_sig_field -= 1

    max_value = (1.0 + max_sig_field * 2.0 ** (-spec.sig_bits)) * (
        2.0 ** (max_exp_field - bias)
    )
    return {
        "bias": bias,
        "max": max_value,
        "min": -max_value,
        "emax": int(math.floor(math.log2(max_value))),
        "smallest_normal": 2.0 ** (1 - bias),
        "smallest_subnormal": 2.0 ** (1 - bias - spec.sig_bits),
    }


def _quantize_mxint(value: float, spec: MXSpec) -> float:
    """Quantize to the OCP MXINT two's-complement element value."""
    scale = 2 ** (spec.element_bits - 2)
    q = np.rint(value * scale)
    q = np.clip(q, -(2 ** (spec.element_bits - 1)), 2 ** (spec.element_bits - 1) - 1)
    return float(q / scale)


def _quantize_float_element(value: float, spec: MXSpec, params: dict) -> float:
    """Round one scaled value to the target MX floating-point element format."""
    if np.isnan(value):
        return np.nan
    if value == 0:
        return 0.0

    sign = -1.0 if value < 0 else 1.0
    value = abs(float(value))

    if np.isinf(value) or value >= params["max"]:
        return sign * params["max"]

    expval = math.floor(math.log2(value))
    expval = max(expval, 1 - params["bias"])
    step = 2.0 ** (expval - spec.sig_bits)
    rounded = np.rint(value / step) * step

    if rounded == 0:
        return 0.0
    if rounded > params["max"]:
        rounded = params["max"]
    return sign * float(rounded)


def _quantize_element(value: float, spec: MXSpec, params: dict) -> float:
    if spec.exp_bits == 0:
        if np.isnan(value):
            return np.nan
        if np.isposinf(value):
            return params["max"]
        if np.isneginf(value):
            return params["min"]
        return _quantize_mxint(value, spec)
    return _quantize_float_element(value, spec, params)


def _resolve_spec(
    format: Union[str, MXSpec, Tuple[int, int]],
    block_size: int,
    scale_exp_bits: Optional[int],
    scale_sig_bits: Optional[int],
) -> MXSpec:
    if isinstance(format, str):
        if format.lower() not in MX_FORMATS:
            raise ValueError(f"Unknown format: {format}")
        spec = MX_FORMATS[format.lower()]
        return replace(
            spec,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits if scale_exp_bits is not None else spec.scale_exp_bits,
            scale_sig_bits=scale_sig_bits if scale_sig_bits is not None else spec.scale_sig_bits,
        )
    if isinstance(format, tuple):
        exp_bits, sig_bits = format
        spec = create_mx_spec(exp_bits, sig_bits, block_size)
        return replace(
            spec,
            scale_exp_bits=scale_exp_bits if scale_exp_bits is not None else spec.scale_exp_bits,
            scale_sig_bits=scale_sig_bits if scale_sig_bits is not None else spec.scale_sig_bits,
        )
    if isinstance(format, MXSpec):
        return replace(
            format,
            scale_exp_bits=scale_exp_bits if scale_exp_bits is not None else format.scale_exp_bits,
            scale_sig_bits=scale_sig_bits if scale_sig_bits is not None else format.scale_sig_bits,
        )
    raise TypeError("format must be str, MXSpec, or tuple")


# ============================================================================
# NumPy Backend: MX Block
# ============================================================================

class MXBlock_:
    """Single OCP MX block quantized with the NumPy reference implementation.
    
    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional block data.
    spec : MXSpec
        MX element and scale specification.
    scale_exp_bits : int, optional
        Override for shared scale exponent bits.
    scale_sig_bits : int, optional
        Override for shared scale significand bits.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        spec: MXSpec,
        scale_exp_bits: Optional[int] = None,
        scale_sig_bits: Optional[int] = None
    ):
        if data.ndim != 1:
            raise ValueError("MXBlock only accepts 1D arrays")
        
        self.spec = spec
        self.size = len(data)
        
        # Override scale bits if provided
        self.scale_exp_bits = scale_exp_bits if scale_exp_bits is not None else spec.scale_exp_bits
        self.scale_sig_bits = scale_sig_bits if scale_sig_bits is not None else spec.scale_sig_bits
        
        self._quantize(data)
    
    def _quantize(self, data: np.ndarray):
        """Quantize data to MX format."""
        params = _mx_float_params(self.spec)

        # Handle all-zero block
        if np.all(data == 0):
            min_scale_exp, _ = _scale_exponent_bounds(self.scale_exp_bits)
            self.shared_scale = 2.0 ** min_scale_exp
            self.quantized_elements = np.zeros(len(data), dtype=np.float32)
            return
        
        # Step 1: Find maximum absolute value for scale
        max_val = np.nanmax(np.abs(data))
        
        if max_val == 0:
            min_scale_exp, _ = _scale_exponent_bounds(self.scale_exp_bits)
            self.shared_scale = 2.0 ** min_scale_exp
            self.quantized_elements = np.zeros(len(data), dtype=np.float32)
            return
        
        # Step 2: Compute the E8M0 shared scale.
        min_scale_exp, max_scale_exp = _scale_exponent_bounds(self.scale_exp_bits)
        if np.isinf(max_val):
            scale_exp = max_scale_exp
        else:
            scale_exp = np.floor(np.log2(max_val)) - params["emax"]
            scale_exp = np.clip(scale_exp, min_scale_exp, max_scale_exp)
        self.shared_scale = 2.0 ** scale_exp
        
        # Step 3: Scale data
        scaled_data = data / self.shared_scale
        
        # Step 4: Quantize each element to MX element format
        self.quantized_elements = self._quantize_elements(scaled_data)
    
    def _quantize_elements(self, data: np.ndarray) -> np.ndarray:
        """
        Quantize elements to MX element format (exp_bits, sig_bits).
        """
        result = np.zeros_like(data, dtype=np.float32)
        params = _mx_float_params(self.spec)

        for i, val in enumerate(data):
            result[i] = _quantize_element(float(val), self.spec, params)
        
        return result
    
    def dequantize(self) -> np.ndarray:
        """Dequantize back to float."""
        return self.quantized_elements * self.shared_scale
    
    def statistics(self) -> dict:
        """Get block statistics."""
        return {
            'size': self.size,
            'shared_scale': self.shared_scale,
            'shared_scale_exp': np.log2(self.shared_scale) if self.shared_scale > 0 else 0,
            'element_range': (self.quantized_elements.min(), self.quantized_elements.max()),
            'bits_per_element': self.spec.element_bits,
            'total_bits': (
                self.spec.element_bits * self.size
                + self.scale_exp_bits
                + self.scale_sig_bits
            ),
        }


# ============================================================================
# NumPy Backend: MX Tensor
# ============================================================================

class MXTensor_:
    """Multi-block OCP MX tensor using the NumPy reference implementation."""
    
    def __init__(
        self,
        data: np.ndarray,
        format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
        block_size: int = 32,
        scale_exp_bits: Optional[int] = None,
        scale_sig_bits: Optional[int] = None
    ):
        self.spec = _resolve_spec(format, block_size, scale_exp_bits, scale_sig_bits)
        
        # Convert to numpy if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        self.original_shape = data.shape
        self.scale_exp_bits = self.spec.scale_exp_bits
        self.scale_sig_bits = self.spec.scale_sig_bits
        
        # Flatten
        data_flat = data.flatten()
        
        # Pad to block size
        block_size = self.spec.block_size
        remainder = len(data_flat) % block_size
        if remainder != 0:
            padding = block_size - remainder
            data_flat = np.pad(data_flat, (0, padding), mode='constant')
        
        self.padded_size = len(data_flat)
        
        # Create blocks (suppress warnings during quantization)
        self.blocks = []
        num_blocks = len(data_flat) // block_size
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            for i in range(num_blocks):
                start_idx = i * block_size
                end_idx = start_idx + block_size
                block_data = data_flat[start_idx:end_idx]
                
                block = MXBlock_(
                    block_data,
                    self.spec,
                    self.scale_exp_bits,
                    self.scale_sig_bits,
                )
                self.blocks.append(block)
    
    def dequantize(self) -> np.ndarray:
        """Dequantize to original shape."""
        dequantized_blocks = [block.dequantize() for block in self.blocks]
        result_flat = np.concatenate(dequantized_blocks)
        result_flat = result_flat[:np.prod(self.original_shape)]
        return result_flat.reshape(self.original_shape)
    
    def statistics(self) -> dict:
        """Get quantization statistics."""
        fp32_bits = np.prod(self.original_shape) * 32
        fp16_bits = np.prod(self.original_shape) * 16
        
        num_blocks = len(self.blocks)
        mx_bits = num_blocks * self.spec.total_bits_per_block
        
        compression_fp32 = fp32_bits / mx_bits
        compression_fp16 = fp16_bits / mx_bits
        
        return {
            'format': self.spec.name,
            'exp_bits': self.spec.exp_bits,
            'sig_bits': self.spec.sig_bits,
            'block_size': self.spec.block_size,
            'scale_exp_bits': self.scale_exp_bits or self.spec.scale_exp_bits,
            'original_shape': self.original_shape,
            'num_blocks': num_blocks,
            'total_elements': np.prod(self.original_shape),
            'compression_ratio_fp32': compression_fp32,
            'compression_ratio_fp16': compression_fp16,
            'bits_per_element': mx_bits / np.prod(self.original_shape),
        }
    
    def __repr__(self):
        stats = self.statistics()
        return (f"MXTensor_(backend=numpy, shape={self.original_shape}, "
                f"format={self.spec.name}, blocks={stats['num_blocks']}, "
                f"compression={stats['compression_ratio_fp16']:.2f}x vs FP16)")


# ============================================================================
# Convenience Functions
# ============================================================================

def mx_quantize(
    data: np.ndarray,
    format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3',
    block_size: int = 32,
    scale_exp_bits: Optional[int] = None,
    scale_sig_bits: Optional[int] = None
) -> np.ndarray:
    """
    NumPy backend: Quantize array to MX format.
    """
    mx_tensor = MXTensor_(data, format, block_size, scale_exp_bits, scale_sig_bits)
    return mx_tensor.dequantize()


def compare_mx_formats(
    data: np.ndarray,
    formats: Optional[List[str]] = None,
    block_size: int = 32
) -> None:
    """Compare different MX formats."""
    if formats is None:
        formats = list(MX_FORMATS.keys())
    
    print("="*100)
    print("MX Format Comparison (NumPy Backend)")
    print("="*100)
    print(f"Input shape: {data.shape}, Total elements: {np.prod(data.shape):,}")
    print("="*100)
    
    header = f"{'Format':<15} {'Element':<10} {'Block':<8} {'Compress':<12} {'MSE':<12} {'MAE':<12}"
    print(header)
    print("-"*100)
    
    for fmt in formats:
        try:
            # Suppress warnings during comparison
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                
                mx = MXTensor_(data, format=fmt, block_size=block_size)
                reconstructed = mx.dequantize()
                
                mse = np.mean((data - reconstructed) ** 2)
                mae = np.mean(np.abs(data - reconstructed))
                
                stats = mx.statistics()
                
                element_fmt = f"E{stats['exp_bits']}M{stats['sig_bits']}"
                row = (f"{stats['format']:<15} "
                       f"{element_fmt:<10} "
                       f"{stats['block_size']:<8} "
                       f"{stats['compression_ratio_fp16']:.2f}x{'':>8} "
                       f"{mse:.2e}{'':>6} "
                       f"{mae:.2e}")
                print(row)
        
        except Exception as e:
            print(f"{fmt:<15} ERROR: {e}")
    
    print("="*100)


__all__ = ['MXBlock_', 'MXTensor_', 'mx_quantize', 'compare_mx_formats']
