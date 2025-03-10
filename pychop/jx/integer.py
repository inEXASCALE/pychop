import jax
import jax.numpy as jnp

class Chopi:
    """
    Integer quantizer.
    
    Parameters
    ----------
    num_bits : int, default=8
        The bitwidth of integer format, the larger it is, the wider range the quantized value can be.

    symmetric : bool, default=False
        Use symmetric quantization (zero_point = 0).

    per_channel : bool, default=False
        Quantize per channel along specified dimension.

    channel_dim : int, default=0
        Dimension to treat as channel axis.

    """
    
    def __init__(self, num_bits=8, symmetric=False, per_channel=False, channel_dim=0):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        self.qmin = -(2 ** (num_bits - 1)) if symmetric else 0
        self.qmax = (2 ** (num_bits - 1)) - 1

        self.scale = None
        self.zero_point = None

        
    def calibrate(self, x):
        """
        Calibrate scale and zero_point based on JAX array.
        
        Parameters
        ----------
        x : Tensor
            Input JAX array to calibrate from.
        """
        if not isinstance(x, jnp.ndarray):
            raise TypeError("Input must be a JAX array")

        if self.per_channel and x.ndim > 1:
            dims = [d for d in range(x.ndim) if d != self.channel_dim]
            if not dims:
                min_val = jnp.min(x)
                max_val = jnp.max(x)
            else:
                min_val = x
                max_val = x
                for d in dims:
                    min_val = jnp.min(min_val, axis=d, keepdims=True)
                    max_val = jnp.max(max_val, axis=d, keepdims=True)
        else:
            min_val = jnp.min(x)
            max_val = jnp.max(x)

        range_val = max_val - min_val
        range_val = jnp.maximum(range_val, 1e-5)
        scale = range_val / (self.qmax - self.qmin)
        zero_point = 0 if self.symmetric else self.qmin - (min_val / scale)

        self.scale = scale
        self.zero_point = zero_point if not self.symmetric else 0


    def quantize(self, x):
        """
        Quantize the array to integers.
        
        Parameters
        ----------
        x : Tensor
            Input array to quantize.
        
        Returns
        ----------
        np.ndarray: Quantized integer array.
        """
        if not isinstance(x, jnp.ndarray):
            raise TypeError("Input must be a JAX array")
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

        q = jnp.round(x / scale + zero_point)
        q = jnp.clip(q, self.qmin, self.qmax)
        return q.astype(jnp.int8)


    def __call__(self, x):
        return self.quantize(x)
    
    
    def dequantize(self, q):
        """
        Dequantize the integer array to floating-point.
        
        Parameters
        ----------
        q : Tensor
            Quantized integer array.
        
        Returns:
        np.ndarray: Dequantized floating-point array.
        """
        if not isinstance(q, jnp.ndarray):
            raise TypeError("Input must be a JAX array")
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

        x = (q - zero_point) * scale
        return x
