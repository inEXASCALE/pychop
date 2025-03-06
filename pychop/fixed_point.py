import os
import torch.nn as nn
from .tch import FPRounding
import torch

class FPoint(object):
    """
    Parameters
    ----------
    x : numpy.ndarray | jax.Array | torch.Tensor,
        The input array. 
    
    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    rmode : int or str, default=1
        The rounding way.
    
    Returns
    ----------
    x_q : numpy.ndarray | jax.Array | torch.Tensor, 
        The quantized array.
    """

    def __init__(self, ibits=4, fbits=4, rmode: str = "nearest",):
        if os.environ['chop_backend'] == 'torch':
            # from .tch import fixed_point
            from .tch import FPRounding
        elif os.environ['chop_backend'] == 'jax':
            # from .jx import fixed_point
            from .jx import FPRounding
        else:
            # from .np import fixed_point
            from .np import FPRounding

        self.rmode = rmode
        self.fpr = FPRounding(ibits, fbits)

    def __call__(self, x):
        return self.fpr.quantize(x, self.rmode)
    




class FQuantizedLayer(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 integer_bits: int,
                 fractional_bits: int,
                 rmode: str = "nearest",
                 bias: bool = True):
        """
        A linear layer with fixed-point quantization for weights, bias, and inputs.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            integer_bits: Number of integer bits (including sign) for Qm.n format
            fractional_bits: Number of fractional bits for Qm.n format
            rmode: Rounding method for quantization
            bias: Whether to include a bias term
        """
        super(FQuantizedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = FPRounding(integer_bits, fractional_bits)
        self.rmode = rmode

        # Initialize weights and bias as floating-point parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed-point quantization.
        
        Args:
            x: Input tensor (batch_size, in_features)
        Returns:
            Output tensor (batch_size, out_features)
        """
        
        return self.quantizer.quantize(x, self.rmode)
