import os, torch
from typing import Optional
from torch.autograd import Function
from .utils import *

class Chopi:
    """
    Front-end wrapper for backend-specific integer quantization implementations.

    This class dispatches to a backend implementation (NumPy, PyTorch, or JAX)
    depending on the environment variable ``chop_backend`` or automatically
    based on the input type when ``chop_backend="auto"``.

    Parameters
    ----------
    bits : int, default=8
        Bit-width of the integer quantization format.

    symmetric : bool, default=False
        If True, use symmetric quantization (zero-point fixed at 0).

    per_channel : bool, default=False
        If True, apply per-channel quantization along the specified axis.

    axis : int, default=0
        Axis to use as the channel dimension when ``per_channel=True``.
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

        self._impl: Optional[object] = None
        self._current_backend: Optional[str] = None

    # ---------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------

    def _resolve_backend(self, X):
        """Resolve backend based on environment and input type."""
        env_backend = os.environ.get("chop_backend", "auto")

        if env_backend == "auto":
            return detect_array_type(X)

        if env_backend not in {"torch", "jax", "numpy"}:
            raise ValueError(
                f"Invalid chop_backend environment value: {env_backend}. "
                "Must be 'torch', 'jax', or 'numpy'."
            )

        return env_backend

    def _get_impl(self, backend: str):
        """Create or reuse backend implementation."""
        if self._current_backend == backend and self._impl is not None:
            return self._impl

        if backend == "torch":
            from .tch.integer import Chopi_ as _ChopiImpl
        elif backend == "jax":
            from .jx.integer import Chopi_ as _ChopiImpl
        elif backend == "numpy":
            from .np.integer import Chopi_ as _ChopiImpl
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                "Must be 'torch', 'jax', or 'numpy'."
            )

        self._impl = _ChopiImpl(
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            axis=self.axis,
        )
        self._current_backend = backend
        return self._impl

    def _ensure_impl(self, X):
        """Ensure correct backend implementation exists."""
        backend = self._resolve_backend(X)

        if self._impl is None or self._current_backend != backend:
            self._get_impl(backend)

        return backend

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def quantize(self, X):
        """
        Quantize input tensor/array.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        Q : array-like
            Quantized representation.
        """
        backend = self._ensure_impl(X)

        if backend == "torch":
            X = to_torch_tensor(X)
        elif backend == "jax":
            X = to_jax_array(X)
        else:
            X = to_numpy_array(X)

        return self._impl.quantize(X)

    def dequantize(self, X):
        """
        Dequantize quantized tensor/array.

        Parameters
        ----------
        X : array-like
            Quantized data.

        Returns
        -------
        D : array-like
            Dequantized tensor/array.
        """
        backend = self._ensure_impl(X)
        return self._impl.dequantize(X)

    def __call__(self, X):
        """
        Quantize and immediately dequantize input.

        Equivalent to:

        >>> self.dequantize(self.quantize(X))

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        array-like
            Dequantized output.
        """
        return self.dequantize(self.quantize(X))

    # ---------------------------------------------------------------------

    @property
    def backend(self) -> Optional[str]:
        """Return current active backend."""
        return self._current_backend


