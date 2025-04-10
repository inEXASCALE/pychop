import torch
import warnings
import torch.nn as nn

class Chopi(nn.Module):
    """
    Integer quantizer with Straight-Through Estimator (STE) for training.
    
    Parameters
    ----------
    bits : int, default=8
        The bitwidth of integer format, the larger it is, the wider range the quantized value can be.

    symmetric : bool, default=False
        Use symmetric quantization (zero_point = 0).

    per_channel : bool, default=False
        Quantize per channel along specified dimension.

    axis : int, default=0
        Dimension to treat as channel axis.
    """
    
    def __init__(self, bits=8, symmetric=False, per_channel=False, axis=0, verbose=False):
        super(Chopi, self).__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.axis = axis

        self.qmin = -(2 ** (bits - 1)) if symmetric else 0
        self.qmax = (2 ** (bits - 1)) - 1

        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(1), requires_grad=False) if not symmetric else 0

        if bits in {8, 16, 32, 64}:
            if bits == 8:
                self.intType = torch.int8
            elif bits == 16:
                self.intType = torch.int16
            elif bits == 32:
                self.intType = torch.int32
            elif bits == 64:
                self.intType = torch.int64
        else:
            warnings.warn("Current int type does not support this bitwidth, use int64 to simulate.")
            self.intType = torch.int64

        self.verbose = verbose

    def calibrate(self, x):
        """
        Calibrate scale and zero_point based on array.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to calibrate from.
        """
        if self.per_channel and x.dim() > 1:
            dims = [d for d in range(x.dim()) if d != self.axis]
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

    def quantize(self, x):
        """
        Quantize the array to integers.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to quantize.
        
        Returns
        ----------
        torch.Tensor: Quantized integer tensor.
        """
        self.calibrate(x)
        if self.per_channel and x.dim() > 1:
            shape = [1] * x.dim()
            shape[self.axis] = -1
            scale = self.scale.view(*shape)
            zero_point = self.zero_point.view(*shape) if not self.symmetric else 0

        else:
            scale = self.scale
            zero_point = self.zero_point if not self.symmetric else 0
            
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, self.qmin, self.qmax)

        if self.verbose:
            clipped = (q == self.qmin).float().mean() + (q == self.qmax).float().mean()
            print(
                f"Quantized range: [{x.min().item():.4f}, {x.max().item():.4f}], Scale: {scale.mean().item():.6f}, Clipped: {clipped.item():.4f}")
        
        return q.to(dtype=self.intType)

    def dequantize(self, q):
        """
        Dequantize the integer tensor to floating-point.
        
        Parameters
        ----------
        q : torch.Tensor
            Quantized integer tensor.
        
        Returns:
        torch.Tensor: Dequantized floating-point tensor.
        """
        if self.per_channel and q.dim() > 1:
            shape = [1] * q.dim()
            shape[self.axis] = -1
            scale = self.scale.view(*shape)
            zero_point = self.zero_point.view(*shape) if not self.symmetric else 0
        else:
            scale = self.scale
            zero_point = self.zero_point if not self.symmetric else 0
        
        x = (q - zero_point) * scale
        return x

    def forward(self, x, training=True):
        """
        Forward pass with quantization and STE during training.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        training : bool, default=True
            Whether in training mode (apply STE) or inference mode.
        
        Returns
        ----------
        torch.Tensor: Quantized and dequantized tensor (training) or quantized tensor (inference).
        """
        if training:
            # Quantize and dequantize
            q = self.quantize(x)
            x_dequant = self.dequantize(q)
            # Apply STE: Pass gradients through x, use quantized values in forward
            if x.requires_grad:
                return x + (x_dequant - x).detach()
            return x_dequant
        
        # Inference: Return quantized values as float32
        return self.quantize(x).to(dtype=torch.float32)
