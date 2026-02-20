import os
import numpy as np

def Bitchop(exp_bits, sig_bits, rmode="nearest_even", subnormal=True, random_state=42, device="cpu", verbose=0):
    """
    Parameters
    ----------
    exp_bits : int
        Number of bits for the exponent in the target format. Determines the range
        of representable values (e.g., 5 bits gives a bias of 15, range -14 to 15).

    sig_bits : int
        Number of bits for the significand (mantissa) in the target format, excluding
        the implicit leading 1 for normalized numbers (e.g., 4 bits allows 0 to 15 plus implicit 1).

    subnormal : boolean
        Whether or not support subnormal numbers are supported.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int or str, default="nearest_even"
        Rounding mode to use when quantizing the significand. Options are:
        - 1 or "nearest_even": Round to nearest value, ties to even (IEEE 754 default).
        - 0 or "nearest_odd": Round to nearest value, ties to odd.
        - 2 or "plus_infinity": Round towards plus infinity (round up).
        - 3 or "minus_infinity": Round towards minus infinity (round down).
        - 4 or "toward_zero": Truncate toward zero (no rounding up).
        - 5 or "stochastic_prop": Stochastic rounding proportional to the fractional part.
        - 6 or "stochastic_equal": Stochastic rounding with 50% probability.
        
    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    device : str or torch.device, optional, default="cpu" 
        Device to perform computations on (e.g., "cpu", "cuda").

    subnormal (bool, optional): If True, supports denormalized numbers (subnormals) when
        the exponent underflows, shifting the significand. If False, underflows result in zero.
        Defaults to True.

    verbose : int | bool, defaul=0
        Whether or not to print out the unit-roundoff.

    Properties
    ----------
    u : float,
        Unit roundoff corresponding to the floating point format

    Methods
    ----------
    Bitchop(x) 
        Method that convert ``x`` to the user-specific arithmetic format.
        
    Returns 
    ----------
    Bitchop | object,
        ``Chop`` instance.

    """

    if os.environ['chop_backend'] == 'torch':
        from .tch.bitchop import Bitchop
        obj = Bitchop(exp_bits=exp_bits, sig_bits=sig_bits, subnormal=subnormal, device=device, 
                   random_state=random_state, rmode=rmode)
        
    elif os.environ['chop_backend'] == 'jax':
        from .jx.bitchop import Bitchop
        obj = Bitchop(exp_bits=exp_bits, sig_bits=sig_bits, subnormal=subnormal, device=device, 
                   random_state=random_state, rmode=rmode)
    else:
        from .np.bitchop import Bitchop
        obj = Bitchop(exp_bits=exp_bits, sig_bits=sig_bits, subnormal=subnormal, random_state=random_state, rmode=rmode)
    
    obj.u = 2**sig_bits / 2
    
    if verbose:
        print("The floating point format is with unit-roundoff of {:e}".format(
            obj.u)+" (≈2^"+str(int(np.log2(obj.u)))+").")
        
    return obj
