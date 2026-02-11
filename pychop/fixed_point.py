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

    rmode : int | str, default=1
        Rounding mode. Supported options:
            - 0 or "nearest_odd": Round to nearest, ties to odd.
            - 1 or "nearest": Round to nearest, ties to even (IEEE 754 default).
            - 2 or "plus_inf": Round towards +∞ (round up).
            - 3 or "minus_inf": Round towards -∞ (round down).
            - 4 or "toward_zero": Truncate toward zero.
            - 5 or "stoc_prop": Stochastic rounding proportional to fractional part.
            - 6 or "stoc_equal": Stochastic rounding with 50% probability.

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
        rmode: int | str = 1,
    ):
        self.ibits = ibits
        self.fbits = fbits
        self.rmode = rmode

        self._impl = None
        self._current_backend = None

    def _get_impl(self, backend: str):
        """Create or reuse the backend-specific implementation."""
        if self._current_backend == backend and self._impl is not None:
            return self._impl

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

    def __call__(self, X):
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

        # Get (or create) the matching implementation
        impl = self._get_impl(target_backend)

        return impl(X)

    def __getattr__(self, name):
        if self._impl is None:
            raise RuntimeError(
                "Chopf backend not yet initialized. "
                "Call the instance with an array/tensor first to determine/create the backend implementation."
            )
        return getattr(self._impl, name)