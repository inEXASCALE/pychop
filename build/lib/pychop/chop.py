import os
import numpy as np

def Chop(prec: str='h', subnormal: bool=None, rmode: int=1, flip: bool=False, explim: int=1, 
         p: float=0.5, randfunc=None, customs=None, random_state: int=0, verbose: int=0):
    """
    Parameters
    ----------
    prec : str, default='h':
        The target arithmetic format.

    subnormal : boolean
       Whether or not to support subnormal numbers.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int or str, default=1
        Rounding mode to use when quantizing the significand. Options are:
        - 1 or "nearest": Round to nearest value, ties to even (IEEE 754 default).
        - 2 or "plus_inf": Round towards plus infinity (round up).
        - 3 or "minus_inf": Round towards minus infinity (round down).
        - 4 or "toward_zero": Truncate toward zero (no rounding up).
        - 5 or "stoc_prop": Stochastic rounding proportional to the fractional part.
        - 6 or "stoc_equal": Stochastic rounding with 50% probability.

    flip : boolean, default=False
        Default is False; If ``flip`` is True, then each element
        of the rounded result has a randomly generated bit in its significand flipped 
        with probability ``p``. This parameter is designed for soft error simulation. 

    explim : boolean, default=True
        Default is True; If ``explim`` is False, then the maximal exponent for
        the specified arithmetic is ignored, thus overflow, underflow, or subnormal numbers
        will be produced only if necessary for the data type.  
        This option is designed for exploring low precisions independent of range limitations.

    p : float, default=0.5
        The probability ``p` for each element of the rounded result has a randomly
        generated bit in its significand flipped  when ``flip`` is True

    randfunc : callable, default=None
        If ``randfunc`` is supplied, then the random numbers used for rounding  will be generated 
        using that function in stochastic rounding (i.e., ``rmode`` of 5 and 6). Default is numbers
        in uniform distribution between 0 and 1, i.e., np.random.uniform.

    customs : dataclass, default=None
        If customs is defined, then use customs.t and customs.emax or (customs.exp_bits and customs.sig_bits) 
        for floating point arithmetic. t is the number of bits in the significand (including the hidden bit) 
        and emax is the maximum value of the exponent customs.exp_bits refers to the exponent bits and sig_bits 
        refers to the significand bits.

    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    verbose : int | bool, defaul=0
        Whether or not to print out the unit-roundoff.

    Properties
    ----------
    u : float,
        Unit roundoff corresponding to the floating point format

    Methods
    ----------
    Chop(x) 
        Method that convert ``x`` to the user-specific arithmetic format.
        
    Returns 
    ----------
    Chop | object,
        ``Chop`` instance.

    """
    rmode_map = {
        0: 0, "nearest_odd": 0,
        1: 1, "nearest": 1,
        2: 2, "plus_inf": 2,
        3: 3, "minus_inf": 3,
        4: 4, "toward_zero": 4,
        5: 5, "stoc_prop": 5,
        6: 6, "stoc_equal": 6,
    }

    try:
        rmode = rmode_map[rmode]
    except KeyError:
        raise NotImplementedError("Invalid parameter for ``rmode``.")
    
    if customs is not None:
        if customs.exp_bits is not None:
            customs.emax = (1 << customs.exp_bits) - 1

        if customs.sig_bits is not None:
            customs.t = customs.sig_bits + 1
    
    if os.environ['chop_backend'] == 'torch':
        from .tch.float_point import Chop

        obj = Chop(prec, subnormal, rmode, flip, explim, p, randfunc, customs, random_state)
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.float_point import Chop

        obj = Chop(prec, subnormal, rmode, flip, explim, p, randfunc, customs, random_state)
    else:
        from .np.float_point import Chop

        obj = Chop(prec, subnormal, rmode, flip, explim, p, randfunc, customs, random_state)
    
    obj.u = 2**(1 - obj.t) / 2
    
    if verbose:
        print("The floating point format is with unit-roundoff of {:e}".format(
            obj.u)+" (≈2^"+str(int(np.log2(obj.u)))+").")
        
    return obj



