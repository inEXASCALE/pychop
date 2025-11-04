import os

def Chopf(ibits: int=4, fbits: int=4, rmode: int =1):
    """
    Fixed-point quantization for numpy.ndarray, jax.Array, and torch.Tensor.
    
    Parameters
    ----------
    ibits : int, default=4
        The bitwidth of integer part. 
    
    fbits : int, default=4
        The bitwidth of fractional part. 
        
    rmode : int or str, default=1
        Rounding mode to use when quantizing the significand. Options are: 
            - 0 or "nearest_odd": Round to nearest value, ties to odd.
            - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
            - 2 or "plus_inf": Round towards plus infinity (round up).
            - 3 or "minus_inf": Round towards minus infinity (round down).
            - 4 or "toward_zero": Truncate toward zero (no rounding up).
            - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
            - 6 or "stoc_equal": Stochastic rounding with 50% probability.

    """
    
    if os.environ['chop_backend'] == 'torch':
        # from .tch import fixed_point
        from .tch import FPRound
    elif os.environ['chop_backend'] == 'jax':
        # from .jx import fixed_point
        from .jx import FPRound
    else:
        # from .np import fixed_point
        from .np import FPRound
    
    return FPRound(ibits, fbits, rmode)
    

    


