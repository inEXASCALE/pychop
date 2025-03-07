import os
import torch.nn as nn
from .tch import FPRound
import torch

class Chopf(object):
    """
    Fixed-point quantization for numpy.ndarray, jax.Array, and torch.Tensor.

    Parameters
    ----------

    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    rmode : int or str, default=1
        The rounding way.


    """

    def __init__(self, ibits=4, fbits=4, rmode: str = "nearest",):
        if os.environ['chop_backend'] == 'torch':
            # from .tch import fixed_point
            from .tch import FPRound
        elif os.environ['chop_backend'] == 'jax':
            # from .jx import fixed_point
            from .jx import FPRound
        else:
            # from .np import fixed_point
            from .np import FPRound

        self.rmode = rmode
        self.fpr = FPRound(ibits, fbits)

    def __call__(self, x):
        """
        x : numpy.ndarray | jax.Array | torch.Tensor,
            The input array. 

        Returns
        ----------
        x_q : numpy.ndarray | jax.Array | torch.Tensor, 
            The quantized array.
        """
        return self.fpr.quantize(x, self.rmode)
    




class FQuantizedLayer(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 ibits: int,
                 fbits: int,
                 rmode: str = "nearest",
                 bias: bool = True):
        """
        A linear layer with fixed-point quantization for weights, bias, and inputs.
            
        Parameters
        ----------
        in_dim : int
            Number of input features
        
        out_dim : int
            Number of output features
        
        ibits : int
            Number of integer bits (including sign) for Qm.n format
        
        fbits : int
            Number of fractional bits for Qm.n format
        
        rmode : int
            Rounding method for quantization
        
        bias : int
            Whether to include a bias term
        """
        super(FQuantizedLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quantizer = Chopf(ibits, fbits)
        self.rmode = rmode

        # Initialize weights and bias as floating-point parameters
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed-point quantization.
        
        Parameters
        ----------
        x : numpy.ndarray | jax.Array | torch.Tensor,
            The input tensor (batch_size, in_dim)

        Returns
        ----------
        Output: numpy.ndarray | jax.Array | torch.Tensor,
            The input tensor (batch_size, out_dim)
        """
        
        return self.quantizer.quantize(x, self.rmode)
