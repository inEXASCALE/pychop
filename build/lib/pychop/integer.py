import os

def Chopi(bits=8, symmetric=False, per_channel=False, axis=0):
    """
    Integer Quantizer: Convert floating point numbers to integers.
    
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
    
    if os.environ['chop_backend'] == 'torch':
        from .tch.integer import Chopi
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.integer import Chopi
        
    else:
        from .np.integer import Chopi


    return Chopi(bits=bits, symmetric=symmetric, per_channel=per_channel, axis=axis)



