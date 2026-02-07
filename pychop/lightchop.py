import os

class LightChop:
    """
    Front-end wrapper class for backend-specific LightChop_ implementations.

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
        - 9 : Round to odd.

    subnormal : boolean, default=True
        Whether or not to support subnormal numbers.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    chunk_size : int, default=800
        the number of elements in each smaller sub-array (or chunk) that a 
        large array is divided into for parallel processing; smaller chunks
        enable more parallelism but increase overhead, while larger chunks 
        reduce overhead but demand more memory. Essentially, chunk size is 
        the granular unit of work Dask manages, balancing 
        computation efficiency and memory constraints. 

    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    verbose : int | bool, defaul=0
        Whether or not to print out the unit-roundoff.


    Returns
    -------
    LightChop_ object that simulates the specified floating-point format and rounding mode.
        The object has an attribute `u` representing the unit roundoff of the simulated floating-point
        format, which is calculated as `2**(1 - t) / 2`, where `t` is the total number of
        bits in the significand (including the hidden bit).        

    
    """

    def __init__(
        self,
        exp_bits: int,
        sig_bits: int,
        rmode: int = 1,
        subnormal: bool = True,
        chunk_size: int = 800,
        random_state: int = 42,
        verbose: int = 0,
    ):
        # select backend
        backend = os.environ.get("chop_backend", "numpy")

        if backend == "torch":
            from .tch.lightchop import LightChop_ as _LightChopImpl
        elif backend == "jax":
            from .jx.lightchop import LightChop_ as _LightChopImpl
        else:
            from .np.lightchop import LightChop_ as _LightChopImpl

        # real implementation
        self._impl = _LightChopImpl(
            exp_bits,
            sig_bits,
            rmode,
            subnormal,
            chunk_size,
            random_state,
        )

        # unit roundoff
        t = sig_bits + 1
        self.u = 2 ** (1 - t) / 2

        # also attach to impl (optional but usually convenient)
        self._impl.u = self.u

        if verbose:

            import numpy as np
            print(
                "The floating point format is with unit-roundoff of {:e}".format(self.u)
                + " (≈2^" + str(int(np.log2(self.u))) + ")."
            )

    def __getattr__(self, name):
        """
        Forward attribute access to backend implementation.
        """
        return getattr(self._impl, name)
