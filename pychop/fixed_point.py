import os

class fpoint(object):
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

    def __init__(self, ibits=4, fbits=4):
        self.ibits = ibits 
        self.fbits = fbits

    def __call__(self, x):
        if os.environ['chop_backend'] == 'torch':
            from .tch import fixed_point
        elif os.environ['chop_backend'] == 'jax':
            from .jx import fixed_point
        else:
            from .np import fixed_point

        return fixed_point.to_fixed_point(x, ibits=self.ibits, fbits=self.fbits)
    

