import os


def Chopi(num_bits=8, symmetric=False, per_channel=False, channel_dim=0):
    """
    Integer Quantizer: Convert floating point numbers to integers.
    
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
    
    if os.environ['chop_backend'] == 'torch':
        from .tch.integer import Chopi
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.integer import Chopi
        
    else:
        from .np.integer import Chopi


    return Chopi(num_bits=num_bits, symmetric=symmetric, per_channel=per_channel, channel_dim=channel_dim)



