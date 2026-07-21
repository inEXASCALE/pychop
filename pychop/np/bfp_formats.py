"""NumPy reference backend for BFP and Flexpoint quantization.

This module implements shared-exponent block quantization with signed
fixed-point mantissas. It is the reference implementation wrapped by the JAX
and TensorFlow BFP front ends.
"""

import numpy as np
import math
from typing import Union, Tuple, Optional, List
from dataclasses import dataclass

# Import shared spec from parent
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bfp_formats import BFPSpec, BFP_FORMATS, create_bfp_spec


def _shared_exponent_bounds(exponent_bits: int) -> Tuple[int, int]:
    bias = 2 ** (exponent_bits - 1) - 1
    return -bias, (2 ** exponent_bits - 2) - bias


# ============================================================================
# NumPy Backend: BFP Block
# ============================================================================

class BFPBlock_:
    """Single BFP block with one shared exponent and signed mantissas."""
    
    def __init__(self, data: np.ndarray, spec: BFPSpec):
        if data.ndim != 1:
            raise ValueError("BFPBlock only accepts 1D arrays")
        
        self.spec = spec
        self.size = len(data)
        self._quantize(data)
    
    def _quantize(self, data: np.ndarray):
        """Quantize data to BFP format."""
        # Handle all-zero block
        if np.all(data == 0):
            min_exp, _ = _shared_exponent_bounds(self.spec.exponent_bits)
            self.shared_exponent = min_exp
            self.mantissas = np.zeros(len(data), dtype=np.float32)
            return
        
        # Find max absolute value
        max_val = np.nanmax(np.abs(data))
        
        if max_val == 0:
            min_exp, _ = _shared_exponent_bounds(self.spec.exponent_bits)
            self.shared_exponent = min_exp
            self.mantissas = np.zeros(len(data), dtype=np.float32)
            return
        
        # Signed p-bit mantissas represent normalized values in
        # [-1, 1 - 2^(1-p)]. Choose the shared exponent so the block maximum
        # does not saturate that fixed-point interval.
        mantissa_levels = 2 ** (self.spec.mantissa_bits - 1)
        max_int = mantissa_levels - 1
        min_int = -mantissa_levels
        max_normalized = max_int / mantissa_levels
        min_exp, max_exp = _shared_exponent_bounds(self.spec.exponent_bits)

        if np.isinf(max_val):
            self.shared_exponent = max_exp
        else:
            exponent = math.ceil(math.log2(max_val / max_normalized))
            self.shared_exponent = int(np.clip(exponent, min_exp, max_exp))
        
        # Normalize by shared exponent
        scale = 2.0 ** self.shared_exponent
        normalized = data / scale
        
        # Quantize mantissas
        quantized_int = np.rint(normalized * mantissa_levels)
        
        # Clip to valid range
        quantized_int = np.clip(quantized_int, min_int, max_int)
        
        # Store as float
        self.mantissas = quantized_int.astype(np.float32) / mantissa_levels
    
    def dequantize(self) -> np.ndarray:
        """Dequantize back to float."""
        scale = 2.0 ** self.shared_exponent
        return self.mantissas * scale


# ============================================================================
# NumPy Backend: BFP Tensor
# ============================================================================

class BFPTensor_:
    """Multi-block BFP tensor using the NumPy reference implementation."""
    
    def __init__(
        self,
        data: np.ndarray,
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
        
        # Convert to numpy if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        self.original_shape = data.shape
        
        # Flatten
        data_flat = data.flatten()
        
        # Pad to block size
        block_size = self.spec.block_size
        remainder = len(data_flat) % block_size
        if remainder != 0:
            padding = block_size - remainder
            data_flat = np.pad(data_flat, (0, padding), mode='constant')
        
        self.padded_size = len(data_flat)
        
        # Create blocks
        self.blocks = []
        num_blocks = len(data_flat) // block_size
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_data = data_flat[start_idx:end_idx]
            
            block = BFPBlock_(block_data, self.spec)
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
        bfp_bits = num_blocks * self.spec.total_bits_per_block
        
        compression_fp32 = fp32_bits / bfp_bits
        compression_fp16 = fp16_bits / bfp_bits
        
        return {
            'format': self.spec.name,
            'mantissa_bits': self.spec.mantissa_bits,
            'block_size': self.spec.block_size,
            'exponent_bits': self.spec.exponent_bits,
            'original_shape': self.original_shape,
            'num_blocks': num_blocks,
            'total_elements': np.prod(self.original_shape),
            'compression_ratio_fp32': compression_fp32,
            'compression_ratio_fp16': compression_fp16,
            'bits_per_element': bfp_bits / np.prod(self.original_shape),
        }
    
    def __repr__(self):
        stats = self.statistics()
        return (f"BFPTensor_(backend=numpy, shape={self.original_shape}, "
                f"format={self.spec.name}, blocks={stats['num_blocks']}, "
                f"compression={stats['compression_ratio_fp16']:.2f}x vs FP16)")


# ============================================================================
# Convenience Functions
# ============================================================================

def bfp_quantize(
    data: np.ndarray,
    format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'
) -> np.ndarray:
    """
    NumPy backend: Quantize array to BFP format.
    
    Parameters
    ----------
    data : np.ndarray
        Input array
    format : str, BFPSpec, or tuple
        BFP format
    
    Returns
    -------
    np.ndarray
        Quantized array (same shape as input)
    """
    bfp_tensor = BFPTensor_(data, format=format)
    return bfp_tensor.dequantize()


__all__ = ['BFPBlock_', 'BFPTensor_', 'bfp_quantize']
