"""
Microscaling (MX) Formats - NumPy Backend Implementation

Pure NumPy implementation of OCP MX formats.
No automatic differentiation, suitable for inference and analysis.

Author: Xinye Chen
"""


import numpy as np
from typing import Union, Tuple, Optional, List
from dataclasses import dataclass
import warnings

# Import shared spec
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mx_formats import MXSpec, MX_FORMATS, create_mx_spec


# ============================================================================
# NumPy Backend: MX Block
# ============================================================================

class MXBlock_:
    """
    NumPy implementation of single MX block.
    
    MX Block structure:
    - Shared scale (exponent) for entire block
    - Each element: sign + exponent + mantissa
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
        # Handle all-zero block
        if np.all(data == 0):
            self.shared_scale = 1.0
            self.quantized_elements = np.zeros(len(data), dtype=np.float32)
            return
        
        # Step 1: Find maximum absolute value for scale
        max_val = np.max(np.abs(data))
        
        if max_val == 0:
            self.shared_scale = 1.0
            self.quantized_elements = np.zeros(len(data), dtype=np.float32)
            return
        
        # Step 2: Compute shared scale (power of 2)
        # Max representable value with given exponent bits
        if self.spec.exp_bits > 0:
            max_exp_val = 2.0 ** (2 ** self.spec.exp_bits - 2)  # Conservative estimate
        else:
            max_exp_val = 2.0 ** 7  # For integer formats
        
        # Find scale such that max_val / scale <= max_exp_val
        if max_val > max_exp_val:
            scale_exp = np.ceil(np.log2(max_val / max_exp_val))
            self.shared_scale = 2.0 ** scale_exp
        else:
            self.shared_scale = 1.0
        
        # Step 3: Scale data
        scaled_data = data / self.shared_scale
        
        # Step 4: Quantize each element to MX element format
        self.quantized_elements = self._quantize_elements(scaled_data)
    
    def _quantize_elements(self, data: np.ndarray) -> np.ndarray:
        """
        Quantize elements to MX element format (exp_bits, sig_bits).
        
        Improved version with overflow handling.
        """
        result = np.zeros_like(data, dtype=np.float32)
        
        # IEEE 754 bias
        bias_ieee = 127
        
        # Target bias for MX format
        if self.spec.exp_bits > 0:
            bias_mx = 2 ** (self.spec.exp_bits - 1) - 1
            max_exp_mx = 2 ** self.spec.exp_bits - 1
        else:
            # Integer format (no exponent)
            bias_mx = 0
            max_exp_mx = 0
        
        for i, val in enumerate(data):
            if val == 0:
                result[i] = 0
                continue
            
            # Handle inf/nan
            if not np.isfinite(val):
                result[i] = val
                continue
            
            # Extract IEEE 754 components
            val_bits = np.float32(val).view(np.uint32)
            sign = (val_bits >> 31) & 1
            exp_ieee = ((val_bits >> 23) & 0xFF)
            mantissa_ieee = val_bits & 0x7FFFFF
            
            # Handle subnormal numbers
            if exp_ieee == 0:
                # Subnormal or zero - flush to zero for simplicity
                result[i] = 0.0
                continue
            
            # Integer format (exp_bits = 0)
            if self.spec.exp_bits == 0:
                # For MXINT8: just round mantissa, keep as normalized float
                if self.spec.sig_bits > 0:
                    shift = 23 - self.spec.sig_bits
                    mantissa_mx = (mantissa_ieee >> shift) << shift
                else:
                    mantissa_mx = 0
                
                result_bits = (sign << 31) | (exp_ieee << 23) | mantissa_mx
                result[i] = np.frombuffer(np.uint32(result_bits).tobytes(), dtype=np.float32)[0]
                continue
            
            # Floating-point format (exp_bits > 0)
            # Convert exponent (with overflow protection)
            exp_ieee_unbiased = int(exp_ieee) - bias_ieee
            
            # Clip exponent to MX range
            exp_mx_unbiased = np.clip(exp_ieee_unbiased, -bias_mx, max_exp_mx - bias_mx - 1)
            exp_mx = exp_mx_unbiased + bias_mx
            
            # Quantize mantissa
            if self.spec.sig_bits > 0:
                shift = 23 - self.spec.sig_bits
                mantissa_mx = (mantissa_ieee >> shift) << shift
            else:
                mantissa_mx = 0
            
            # Reconstruct IEEE 754 (with overflow protection)
            try:
                new_exp_ieee = int(exp_mx - bias_mx + bias_ieee)
                # Clip to valid IEEE 754 range
                new_exp_ieee = np.clip(new_exp_ieee, 0, 254)
                
                result_bits = (sign << 31) | (new_exp_ieee << 23) | mantissa_mx
                result[i] = np.frombuffer(np.uint32(result_bits).tobytes(), dtype=np.float32)[0]
            except (OverflowError, ValueError):
                # If still overflow, use original value
                result[i] = val
        
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
            'total_bits': self.spec.element_bits * self.size + self.scale_exp_bits,
        }


# ============================================================================
# NumPy Backend: MX Tensor
# ============================================================================

class MXTensor_:
    """
    NumPy implementation of multi-block MX tensor.
    """
    
    def __init__(
        self,
        data: np.ndarray,
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
        
        # Convert to numpy if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        self.original_shape = data.shape
        self.scale_exp_bits = scale_exp_bits
        self.scale_sig_bits = scale_sig_bits
        
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
                
                block = MXBlock_(block_data, self.spec, scale_exp_bits, scale_sig_bits)
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