import torch
import torch.nn as nn
import torch.nn.functional as F

class Chopi(nn.Module):
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
        super(Chopi, self).__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        self.qmin = -(2 ** (num_bits - 1)) if symmetric else 0
        self.qmax = (2 ** (num_bits - 1)) - 1

        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(1), requires_grad=False) if not symmetric else 0

    
    def calibrate(self, x):
        """
        Calibrate scale and zero_point based on array.
        
        Parameters
        ----------
        x : np.ndarray
            Input array to calibrate from.
        """
        if self.per_channel and x.dim() > 1:
            dims = [d for d in range(x.dim()) if d != self.channel_dim]
            min_val = x
            max_val = x
            for d in dims:
                min_val = min_val.min(dim=d, keepdim=True)[0]
                max_val = max_val.max(dim=d, keepdim=True)[0]
        else:
            min_val = x.min()
            max_val = x.max()

        range_val = max_val - min_val
        range_val = range_val.clamp(min=1e-5)
        scale = range_val / (self.qmax - self.qmin)
        zero_point = 0 if self.symmetric else self.qmin - (min_val / scale)

        self.scale.data = scale.detach()
        if not self.symmetric:
            self.zero_point.data = zero_point.detach()

    def __call__(self, x):
        return self.quantize(x)

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
        self.calibrate(x)
        if self.per_channel and x.dim() > 1:
            shape = [1] * x.dim()
            shape[self.channel_dim] = -1
            scale = self.scale.view(*shape)
            zero_point = self.zero_point.view(*shape) if not self.symmetric else 0
        else:
            scale = self.scale
            zero_point = self.zero_point if not self.symmetric else 0
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, self.qmin, self.qmax)
        return q
        

    def dequantize(self, q):
        """
        Dequantize the integer NumPy array to floating-point.
        
        Parameters
        ----------
        q : Tensor
            Quantized integer NumPy array.
        
        Returns:
        np.ndarray: Dequantized floating-point array.
        """
        if self.per_channel and q.dim() > 1:
            shape = [1] * q.dim()
            shape[self.channel_dim] = -1
            scale = self.scale.view(*shape)
            zero_point = self.zero_point.view(*shape) if not self.symmetric else 0
        else:
            scale = self.scale
            zero_point = self.zero_point if not self.symmetric else 0
        x = (q - zero_point) * scale
        return x
        

    def forward(self, x, training=True):
        if training:
            q = self.quantize(x)
            return self.dequantize(q)
        return self.quantize(x)


