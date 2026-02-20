import os
from .utils import detect_array_type, to_numpy_array, to_torch_tensor, to_jax_array


class Chopf:
    """
    Front-end wrapper class for backend-specific fixed-point quantization implementations.

    Parameters
    ----------
    ibits : int, default=4
        Bitwidth of the integer part (including sign bit if signed).

    fbits : int, default=4
        Bitwidth of the fractional part.

    rmode : int, default=1
        Rounding mode. Supported options:
            - 0: Round to nearest, ties to odd.
            - 1: Round to nearest, ties to even (IEEE 754 default).
            - 2: Round towards +∞ (round up).
            - 3: Round towards -∞ (round down).
            - 4: Truncate toward zero.
            - 5: Stochastic rounding proportional to fractional part.
            - 6: Stochastic rounding with 50% probability.

    Returns
    -------
    Chopf object that can be called on arrays/tensors to quantize them to the specified
    fixed-point format. The backend is automatically detected from the input type when
    ``os.environ["chop_backend"]`` is "auto" (default), or fixed via the environment
    variable.
    """

    def __init__(
        self,
        ibits: int = 4,
        fbits: int = 4,
        rmode: int = 1,
    ):
        self.ibits = ibits
        self.fbits = fbits
        self.rmode = rmode

        self._impl = None
        self._current_backend = None
        backend = os.environ.get("chop_backend", "auto")
        self._get_impl(backend)

    def _get_impl(self, backend: str):
        """Create or reuse the backend-specific implementation."""
        if self._current_backend == backend and self._impl is not None:
            return self._impl

        impl = None
        if backend != "auto":
            if backend == "torch":
                from .tch import FPRound_ as _ChopfImpl
            elif backend == "jax":
                from .jx import FPRound_ as _ChopfImpl
            elif backend == "numpy":
                from .np import FPRound_ as _ChopfImpl
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. Must be 'torch', 'jax', or 'numpy'."
                )

            impl = _ChopfImpl(
                ibits=self.ibits,
                fbits=self.fbits,
                rmode=self.rmode,
            )

        self._impl = impl
        self._current_backend = backend
        return impl

    def quantize(self, X):
        """
        Quantize input to fixed-point format.
        Automatically initializes the backend on first call (fixes the common
        "backend not yet initialized" error in layers).
        """
        if self._impl is None:
            env_backend = os.environ.get("chop_backend", "auto")
            if env_backend == "auto":
                target_backend = detect_array_type(X)
            else:
                target_backend = env_backend
                if target_backend not in {"torch", "jax", "numpy"}:
                    raise ValueError(
                        f"Invalid chop_backend environment value: {target_backend}. "
                        "Must be 'torch', 'jax', or 'numpy'."
                    )

            # Convert input to the target backend type
            if target_backend == "torch":
                X = to_torch_tensor(X)
            elif target_backend == "jax":
                X = to_jax_array(X)
            else:  # numpy
                X = to_numpy_array(X)

            # Create the backend implementation
            self._get_impl(target_backend)

        return self._impl.quantize(X)


    def __call__(self, X):
        """Direct call interface (keeps backward compatibility)."""
        return self.quantize(X)

    def __getattr__(self, name):
        if self._impl is None:
            raise RuntimeError(
                "Chopf backend not yet initialized. "
                "Call .quantize() or the Chopf instance first "
                "to determine/create the backend implementation."
            )
        return getattr(self._impl, name)