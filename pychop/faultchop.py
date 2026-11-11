"""Backend-dispatched floating-point chopping with optional bit-flip faults.

``FaultChop`` selects the NumPy, PyTorch, JAX, or TensorFlow backend and
combines floating-point quantization with optional significand bit flips for
soft-error simulation.
"""

import os
from .utils import detect_array_type, to_numpy_array, to_torch_tensor, to_jax_array, to_tensorflow_tensor
from ._fault_formats import FORMAT_PARAMS, resolve_customs

class FaultChop:
    """
    Full-featured research interface for low-precision arithmetic with soft-error simulation
    (inspired by Nick Higham's classic chop).

    This class provides complete control over precision, rounding, and fault injection,
    making it ideal for numerical robustness studies and fault-tolerant computing research.

    Front-end wrapper class for backend-specific Chop_ implementations.

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
        - 10 or "cadna": Stochastic rounding - CADNA-style random directed rounding.

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

    overflow : {'saturate', 'nan'}, default='saturate'
        Overflow handling for OCP ``fp8-e4m3``. Saturation returns ±448;
        ``'nan'`` returns NaN for values outside the finite range.

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

    def __init__(
        self,
        prec: str = "h",
        subnormal: bool = None,
        rmode: int = 1,
        flip: bool = False,
        explim: int = 1,
        p: float = 0.5,
        randfunc=None,
        customs=None,
        random_state: int = 0,
        verbose: int = 0,
        overflow: str = "saturate",
    ):
        # -------- rmode mapping --------
        rmode_map = {
            0: 0, "nearest_odd": 0,
            1: 1, "nearest": 1,
            2: 2, "plus_inf": 2,
            3: 3, "minus_inf": 3,
            4: 4, "toward_zero": 4,
            5: 5, "stoc_prop": 5,
            6: 6, "stoc_equal": 6,
            10: 10, "cadna": 10,
        }

        try:
            rmode = rmode_map[rmode]
        except KeyError:
            raise NotImplementedError("Invalid parameter for ``rmode``.")

        # A zero fault probability must leave the rounded value unchanged.
        flip = bool(flip and p != 0)

        # -------- customs preprocessing --------
        if customs is not None:
            customs = resolve_customs(customs)

        if overflow not in ("saturate", "nan"):
            raise ValueError("overflow must be 'saturate' or 'nan'")
        if overflow != "saturate" and (prec != "fp8-e4m3" or customs is not None):
            raise ValueError("overflow='nan' is available only for fp8-e4m3")
        if prec == "fp8-e4m3" and customs is None and (rmode != 1 or flip or not explim):
            raise ValueError(
                "fp8-e4m3 supports nearest rounding with exponent limits and no bit flips; "
                "use q43 for legacy modes"
            )

        # -------- unit roundoff --------
        if customs:
            self.t, self.emax = customs.t, customs.emax
        else:
            if prec not in FORMAT_PARAMS:
                raise ValueError('Invalid prec value')
            self.t, self.emax = FORMAT_PARAMS[prec]

        self.u = 2 ** (1 - self.t) / 2
        self._impl = None
        self.prec = prec
        self.rmode = rmode
        self.subnormal = subnormal
        self.flip = flip
        self.explim = explim
        self.p = p
        self.randfunc = randfunc
        self.customs = customs
        self.random_state = random_state
        self.overflow = overflow
        

        # -------- backend selection --------
        backend = os.environ.get("chop_backend", "auto")
        self.verbose = verbose

        if backend != "auto":
            if backend == "torch":
                from .tch.float_point import Chop_ as _ChopImpl
            elif backend == "jax":
                from .jx.float_point import Chop_ as _ChopImpl
            elif backend == "tensorflow":
                from .tf.float_point import Chop_ as _ChopImpl
            elif backend == "numpy":
                from .np.float_point import Chop_ as _ChopImpl

            self._impl = _ChopImpl(
                prec,
                subnormal,
                rmode,
                flip,
                explim,
                p,
                randfunc,
                customs,
                random_state,
            )

            self._impl.u = self.u
            self._impl.overflow = overflow
            if backend == "tensorflow":
                self._impl._np_impl.overflow = overflow

        if self.verbose:
            import numpy as np
            print(
                "The floating point format is with unit-roundoff of {:e}".format(self.u)
                + " (≈2^" + str(int(np.log2(self.u))) + ")."
            )


    def __call__(self, X):
        if os.environ.get('chop_backend', 'auto') == 'auto':
             # sanity check for supported array types
            backend =  detect_array_type(X, verbose=self.verbose) 
            self._impl = None 

            if backend == "torch":
                X = to_torch_tensor(X)  
                from .tch.float_point import Chop_ as _ChopImpl
            elif backend == "jax":
                X = to_jax_array(X)  
                from .jx.float_point import Chop_ as _ChopImpl
            elif backend == "tensorflow":
                X = to_tensorflow_tensor(X)
                from .tf.float_point import Chop_ as _ChopImpl
            else:
                X = to_numpy_array(X)  
                from .np.float_point import Chop_ as _ChopImpl

            self._impl = _ChopImpl(
                self.prec,
                self.subnormal,
                self.rmode,
                self.flip,
                self.explim,
                self.p,
                self.randfunc,
                self.customs,
                self.random_state,
            )
            self._impl.overflow = self.overflow
            if backend == "tensorflow":
                self._impl._np_impl.overflow = self.overflow

        return self._impl(X)


    def __getattr__(self, name):
        """
        Forward attribute access to backend implementation.
        """
        if self._impl is None:
            print(f"""Warning: LightChop backend not yet determined."""
                  f"""Call the LightChop instance with an array to determine the backend and initialize the implementation.""")

        return getattr(self._impl, name)


