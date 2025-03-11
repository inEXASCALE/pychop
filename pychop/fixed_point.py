import os
from .tch import FPRound

class Chopf(object):
    """
    Fixed-point quantization for numpy.ndarray, jax.Array, and torch.Tensor.

    Parameters
    ----------
    ibits: int, default=4
        The bitwidth of integer part. 

    fbits: int, default=4
        The bitwidth of fractional part. 
        
    rmode: int or str, default=1
            Rounding mode to use when quantizing the significand. Options are: 
                - 0 or "nearest_odd": Round to nearest value, ties to odd.
                - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
                - 2 or "plus_inf": Round towards plus infinity (round up).
                - 3 or "minus_inf": Round towards minus infinity (round down).
                - 4 or "toward_zero": Truncate toward zero (no rounding up).
                - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
                - 6 or "stoc_equal": Stochastic rounding with 50% probability.

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
        x: numpy.ndarray | jax.Array | torch.Tensor,
            The input array. 

        Returns
        ----------
        x_q: numpy.ndarray | jax.Array | torch.Tensor, 
            The quantized array.
        """
        return self.fpr.quantize(x, self.rmode)
    


