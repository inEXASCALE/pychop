"""
Parameters
----------
bits : int, default=8
    The bitwidth of integer format, the larger it is, the wider range the quantized value can be.
    
sign : bool, default=1
    Whether or not to quantize the value to symmetric integer range.

zpoint : bool, default=1
    Whether or not to set the zero point to zero. If `zpoint=0`, then the quantized range must be symmetric.
    
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
    
"""


import os

def quant(bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
    if os.environ['chop_backend'] == 'numpy':
        from .np.quant import quant
        return quant(bits, sign, zpoint, rd_func, clip_range, epsilon)

    else:
        from .tch.quant import quant
        return quant(bits, sign, zpoint, rd_func, clip_range, epsilon)
    