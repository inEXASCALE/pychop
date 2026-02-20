import os
from .utils import detect_array_type, to_numpy_array, to_torch_tensor, to_jax_array


class Chopi:
    """
    Front-end wrapper class for backend-specific integer quantization implementations.

    Parameters
    ----------
    bits : int, default=8
        The bitwidth of the integer format. Larger values allow a wider range.

    symmetric : bool, default=False
        If True, use symmetric quantization (zero-point fixed at 0).

    per_channel : bool, default=False
        If True, perform quantization per channel along the specified axis.

    axis : int, default=0
        The axis to treat as the channel dimension when per_channel=True.

    Returns
    -------
    Chopi object that can be called on arrays/tensors to quantize them to the specified
    integer format. The backend is automatically detected from the input type when
    ``os.environ["chop_backend"]`` is "auto" (default), or fixed via the environment
    variable.
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        per_channel: bool = False,
        axis: int = 0,
    ):
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.axis = axis

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
                from .tch.integer import Chopi_ as _ChopiImpl
            elif backend == "jax":
                from .jx.integer import Chopi_ as _ChopiImpl
            elif backend == "numpy":
                from .np.integer import Chopi_ as _ChopiImpl
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. Must be 'torch', 'jax', or 'numpy'."
                )
            
            impl = _ChopiImpl(
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                axis=self.axis,
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
                "Chopi backend not yet initialized. "
                "Call the instance with an array/tensor first to determine/create the backend implementation."
            )
        return getattr(self._impl, name)