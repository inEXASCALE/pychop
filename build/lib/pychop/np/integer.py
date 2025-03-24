import numpy as np
import warnings

class Chopi:
    def __init__(self, num_bits=8, symmetric=False, per_channel=False, channel_dim=0):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.qmin = -(2 ** (num_bits - 1)) if symmetric else 0
        self.qmax = (2 ** (num_bits - 1)) - 1
        self.scale = None
        self.zero_point = None

        if num_bits in {8, 16, 32, 64}:
            if num_bits == 8:
                self.intType = np.int8
                
            elif num_bits == 16:
                self.intType = np.int16
                
            elif num_bits == 32:
                self.intType = np.int32
                
            elif num_bits == 64:
                self.intType = np.int64
        else:
            warnings.warn("Current int type not support this bitwidth, use int64 to simulate.")
            self.intType = np.int64
            
    def calibrate(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if self.per_channel and x.ndim > 1:
            dims = [d for d in range(x.ndim) if d != self.channel_dim]
            if not dims:
                min_val = np.min(x)
                max_val = np.max(x)
            else:
                min_val = x
                max_val = x
                for d in dims:
                    min_val = np.min(min_val, axis=d, keepdims=True)
                    max_val = np.max(max_val, axis=d, keepdims=True)
        else:
            min_val = np.min(x)
            max_val = np.max(x)
            
        range_val = max_val - min_val
        range_val = np.maximum(range_val, 1e-5)
        scale = range_val / (self.qmax - self.qmin)
        zero_point = 0 if self.symmetric else np.round(self.qmin - (min_val / scale)).astype(np.int32)
        self.scale = scale
        self.zero_point = zero_point if not self.symmetric else 0


    def __call__(self, x):
        return self.quantize(x)

    
    def quantize(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if self.scale is None or (not self.symmetric and self.zero_point is None):
            self.calibrate(x)
        if self.per_channel and x.ndim > 1:
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            scale = self.scale.reshape(*shape)
            zero_point = self.zero_point.reshape(*shape) if not self.symmetric else 0
        else:
            scale = self.scale
            zero_point = self.zero_point if not self.symmetric else 0
        q = np.round(x / scale + zero_point).clip(self.qmin, self.qmax).astype(self.intType)

        return q

    def dequantize(self, q):
        if not isinstance(q, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if self.scale is None or (not self.symmetric and self.zero_point is None):
            raise ValueError("Quantizer must be calibrated before dequantization")
        if self.per_channel and q.ndim > 1:
            shape = [1] * q.ndim
            shape[self.channel_dim] = -1
            scale = self.scale.reshape(*shape)
            zero_point = self.zero_point.reshape(*shape) if not self.symmetric else 0
        else:
            scale = self.scale
            zero_point = self.zero_point if not self.symmetric else 0
        return (q - zero_point) * scale