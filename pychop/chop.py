import os
import numpy as np

def chop(prec='h', subnormal=None, rmode=1, flip=False, explim=1, device='cpu',
         p=0.5, randfunc=None, customs=None, random_state=0, verbose=0):
    """
    Parameters
    ----------
    prec : str, default='s':
        The target arithmetic format.

    subnormal : boolean
       Whether or not to support subnormal numbers.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int, default=1
        The supported rounding modes include:
            1. Round to nearest using round to even last bit to break ties (the default).
            2. Round towards plus infinity (round up).
            3. Round towards minus infinity (round down).
            4. Round towards zero.
            5. Stochastic rounding 
                round to the next larger or next smaller floating-point number
                with probability proportional to the distance to those floating-point numbers.
            6. Stochastic rounding
                round to the next larger or next smaller 
                floating-point number with equal probability.

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
        If customs is defined, then use customs.t and customs.emax for floating point arithmetic.

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
    chop(x) 
        Method that convert ``x`` to the user-specific arithmetic format.
        
    Returns 
    ----------
    chop | object,
        ``chop`` instance.

    """

    if os.environ['chop_backend'] == 'torch':
        from .tch.chop import chop

        obj = chop(prec, subnormal, rmode, flip, 
                explim, device, p, randfunc, customs, random_state)
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.chop import chop

        obj = chop(prec, subnormal, rmode, flip, 
                explim, p, randfunc, customs, random_state)
    else:

        from .np.chop import chop

        obj = chop(prec, subnormal, rmode, flip, 
                explim, p, randfunc, customs, random_state)
    
    obj.u = 2**(1 - obj.t) / 2
    
    if verbose:
        print("The floating point format is with unit-roundoff of {:e}".format(
            obj.u)+" (≈2^"+str(int(np.log2(obj.u)))+").")
        
    return obj
