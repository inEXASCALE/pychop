import os

def to_fixed_point(x, ibits=4, fbits=4):
    """
    Parameters
    ----------
    x : numpy.ndarray | jax.Array | torch.Tensor,
        The input array.    
    
    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    Returns
    ----------
    x_q : numpy.ndarray | jax.Array | torch.Tensor, 
        The quantized array.
    """
    if os.environ['chop_backend'] == 'torch':
        from .tch import fixed_point
    elif os.environ['chop_backend'] == 'jax':
        from .jx import fixed_point
    else:
        from .np import fixed_point
        
    return fixed_point.to_fixed_point(x, ibits=4, fbits=4)