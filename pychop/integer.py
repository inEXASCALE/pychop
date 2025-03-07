import os
import torch


def quant(bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
    """
    Parameters
    ----------
    bits : int, default=8
        The bitwidth of integer format, the larger it is, the wider range the quantized value can be.
        
    sign : bool, default=1
        Whether or not to quantize the value to symmetric integer range.

    zpoint : bool, default=1
        Whether or not to compute the zero point. If `zpoint=0`, then the quantized range must be symmetric.
        
    rd_func : function, default=None
        The rounding function used for the quantization. The default is round to nearest.
        
    clip_range : list, default=None
        The clipping function for the quantization.
        
    epsilon : double, default=1e-12
        When the x is comprised of single value, then the scaling factor will be (b - a + epsilon) / (alpha - beta)
        for mapping [a, b] to [alpha, beta].
        

    Methods
    ----------
    quant(x):
        Method that quantize ``x`` to the user-specific arithmetic format.

        
    Returns
    ----------  
    quant | object,
        ``quant`` instance.
        
    """
    if os.environ['chop_backend'] == 'torch':
        from .tch.quant import quant
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.quant import quant
        
    else:
        from .np.quant import quant
    
    return quant(bits, sign, zpoint, rd_func, clip_range, epsilon)
    


class Chopi(object):
    """
    Parameters
    ----------
    bits : int, default=8
        The bitwidth of integer format, the larger it is, the wider range the quantized value can be.
        
    sign : bool, default=1
        Whether or not to quantize the value to symmetric integer range.

    zpoint : bool, default=1
        Whether or not to compute the zero point. If `zpoint=0`, then the quantized range must be symmetric.
        
    rd_func : function, default=None
        The rounding function used for the quantization. The default is round to nearest.
        
    clip_range : list, default=None
        The clipping function for the quantization.
        
    epsilon : double, default=1e-12
        When the x is comprised of single value, then the scaling factor will be (b - a + epsilon) / (alpha - beta)
        for mapping [a, b] to [alpha, beta].
        

    """
    def __init__(self, bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
        self.bits = bits
        self.sign = sign
        self.zpoint = zpoint
        self.rd_func = rd_func
        self.clip_range = clip_range
        self.epsilon = epsilon
        

    def __call__(self):
        return self.quant(bits=self.bits, 
                          sign=self.sign, 
                          zpoint=self.zpoint, 
                          rd_func=self.rd_func, 
                          clip_range=self.clip_range, 
                          epsilon=self.epsilon
                          )


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
        