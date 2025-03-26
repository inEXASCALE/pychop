import os

def LightChop(exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool=True, chunk_size: int=1000, random_state: int=42):
    """

    Parameters
    ----------
    exp_bits : int, 
        Bitwidth for exponent of binary floating point numbers.

    sig_bits: int,
        Bitwidth for significand of binary floating point numbers.
        
    rmode : int, default=1
        Rounding mode to use when quantizing the significand. Options are:
        - 1 : Round to nearest value, ties to even (IEEE 754 default).
        - 2 : Round towards plus infinity (round up).
        - 3 : Round towards minus infinity (round down).
        - 4 : Truncate toward zero (no rounding up).
        - 5 : Stochastic rounding proportional to the fractional part.
        - 6 : Stochastic rounding with 50% probability.
        - 7 : Round to nearest value, ties to zero.
        - 8 : Round to nearest value, ties to away.

    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    """
    
    if os.environ['chop_backend'] == 'torch':
        from .tch.lightchop import LightChop
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.lightchop import LightChop
        
    else:
        from .np.lightchop import LightChop

    return LightChop(exp_bits, sig_bits, rmode, subnormal, chunk_size, random_state)


