import torch
from .lightchop import LightChop
from .fixed_point import Chopf
from .integer import Chopi
import torch.nn as nn


class QuantizedLayer(torch.nn.Module):
    """Example of a quantized linear layer"""
    def __init__(self, 
                 exp_bits: int,
                 sig_bits: int,
                 rmode: str = "nearest"):
        
        super().__init__()
        self.quantizer = LightChop(exp_bits, sig_bits, rmode)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(x)


class IntQuantizedLayer(torch.nn.Module):
    """
    __init__(config)
        Apply ``pychop`` to quantization aware training, 
        One can feed [quant | chop | fixed_point] module as base for quantization.

    """

    def __init__(self, bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
        super(IntQuantizedLayer, self).__init__()
        self.chopi = Chopi(bits=bits, sign=sign, zpoint=zpoint, rd_func=rd_func, clip_range=clip_range, epsilon=epsilon)
        
    def forward(self, x):
        return self.chopi(x)
        

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
            Rounding mode to use when quantizing the significand. Options are:
            - 0 or "nearest_odd": Round to nearest value, ties to odd.
            - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
            - 2 or "plus_inf": Round towards plus infinity (round up).
            - 3 or "minus_inf": Round towards minus infinity (round down).
            - 4 or "toward_zero": Truncate toward zero (no rounding up).
            - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
            - 6 or "stoc_equal": Stochastic rounding with 50% probability.

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
