"""Backend-agnostic floating-point quantization front end.

``Chop`` dispatches to NumPy, PyTorch, JAX, or TensorFlow ``LightChop_``
implementations based on ``pychop.backend`` or input-array auto detection.
It supports deterministic IEEE-style rounding modes, stochastic rounding, and
CADNA-style random directed rounding.
"""

import os
from .utils import detect_array_type, to_numpy_array, to_torch_tensor, to_jax_array, to_tensorflow_tensor


class Chop:
    """
    Front-end wrapper class for backend-specific LightChop_ implementations.

    Parameters
    ----------
    exp_bits : int, 
        Bitwidth for exponent of binary floating point numbers.

    sig_bits: int,
        Number of trailing fraction bits, excluding the implicit leading bit.
        
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
        - 10 : CADNA-style random directed rounding.

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

    random_state : int, default=42
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
    
        # unit roundoff
        t = sig_bits + 1
        self.u = 2 ** (1 - t) / 2
        self._impl = None
        self._impl_backend = None

        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.subnormal = subnormal
        self.chunk_size = chunk_size
        self.random_state = random_state

        # select backend
        backend = os.environ.get("chop_backend", "auto")
        self.verbose = verbose

        if backend != "auto":
            self._get_impl(backend)

        if self.verbose:
            import numpy as np
            print(
                "The floating point format is with unit-roundoff of {:e}".format(self.u)
                + " (≈2^" + str(int(np.log2(self.u))) + ")."
            )


    def _get_impl(self, backend):
        if backend == "torch":
            from .tch.lightchop import LightChop_ as _LightChopImpl
        elif backend == "jax":
            from .jx.lightchop import LightChop_ as _LightChopImpl
        elif backend == "tensorflow":
            from .tf.lightchop import LightChop_ as _LightChopImpl
        elif backend == "numpy":
            from .np.lightchop import LightChop_ as _LightChopImpl
        else:
            raise ValueError(f"Unsupported backend: {backend!r}")

        self._impl = _LightChopImpl(
            self.exp_bits,
            self.sig_bits,
            self.rmode,
            self.subnormal,
            self.chunk_size,
            self.random_state,
        )
        self._impl.u = self.u
        self._impl_backend = backend


    def __call__(self, X):
        backend_env = os.environ.get('chop_backend', 'auto')
        if backend_env == 'auto':
            # sanity check for supported array types
            backend = detect_array_type(X, verbose=self.verbose)
            if backend in ("list", "unknown"):
                import numpy as np
                X = np.asarray(X)
                backend = "numpy"
            if self._impl is None or self._impl_backend != backend:
                self._get_impl(backend)
            if backend == "torch":
                X = to_torch_tensor(X)
            elif backend == "jax":
                X = to_jax_array(X)
            elif backend == "tensorflow":
                X = to_tensorflow_tensor(X)
            else:
                X = to_numpy_array(X)
        elif self._impl is None or self._impl_backend != backend_env:
            self._get_impl(backend_env)

        if self._impl is None:
            raise RuntimeError("Chop backend implementation was not initialized.")

        return self._impl(X)


    def __getattr__(self, name):
        """
        Forward attribute access to backend implementation.
        """
        if self._impl is None:
            print(f"""Warning: LightChop backend not yet determined."""
                  f"""Call the LightChop instance with an array to determine the backend and initialize the implementation.""")

        return getattr(self._impl, name)
