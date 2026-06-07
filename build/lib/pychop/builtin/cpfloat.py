"""
pychop.builtin.cpfloat
======================

This module implements :class:`~pychop.builtin.cpfloat.CPFloat`, a scalar wrapper
that preserves *chopped precision* semantics.

`CPFloat` is **backend-aware**: it feeds scalars into the active `pychop.get_backend()`
array/tensor type (NumPy / Torch / JAX) before calling the backend-specific
`Chop` implementation. This avoids type mismatches such as passing a NumPy array
to a Torch chopper.

The API and docstrings follow a scikit-learn style: clear contracts, small
surface area, and explicit notes about limitations.

Notes
-----
- NumPy ufunc support is provided via ``__array_ufunc__``.
- Python's built-in ``math`` module will typically coerce inputs to ``float``
  and therefore does not preserve chopped semantics automatically. Prefer NumPy
  ufuncs (``np.sin``, ``np.sqrt``, ...) or create wrappers if needed.
- This class is designed to be used as an output type for chopped scalar
  results (e.g., `det`, `norm`, `trace`), and for chaining scalar arithmetic
  while keeping chopped semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pychop


Number = Union[int, float, complex, np.number]


@dataclass(frozen=False)
class CPFloat:
    """
    Chopped-precision scalar.

    Parameters
    ----------
    value : int, float, complex, or numpy scalar
        Input scalar value. It is chopped immediately at construction time.
    chopper : callable
        A `Chop`-like object. It must be callable and accept a backend-appropriate
        scalar container:
        - NumPy backend: expects a NumPy array/scalar
        - Torch backend: expects a `torch.Tensor`
        - JAX backend: expects a `jax.Array`

    Attributes
    ----------
    value : float
        The chopped value stored as a Python scalar (typically `float`, but can
        be complex depending on backend and operation).
    chopper : callable
        The chopping/quantization operator.

    Examples
    --------
    Basic arithmetic stays chopped::

        from pychop import Chop
        from pychop.builtin import CPFloat

        half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

        a = CPFloat(1.234567, half)
        b = CPFloat(0.987654, half)

        c = a + b
        d = a * b / 2.0 - 0.1

    NumPy ufunc interoperability::

        import numpy as np
        x = CPFloat(1.234, half)

        y = np.sin(x)          # -> CPFloat
        z = np.sqrt(x + 1.0)   # -> CPFloat

    Notes
    -----
    - Mixed-chopper operations are disallowed for CPFloat binary arithmetic and
      NumPy ufuncs.
    - For torch/jax specific functions (e.g., torch.sin), CPFloat will be
      coerced to a Python float unless you build explicit wrappers.
    """

    value: Any
    chopper: Any

    def __init__(self, value: Number, chopper: Any):
        self.chopper = chopper
        self.value = self._chop_scalar(value)

    # ---------------------------------------------------------------------
    # Backend-aware chopping
    # ---------------------------------------------------------------------
    def _chop_scalar(self, val: Any):
        """
        Chop a scalar using the active backend.

        Parameters
        ----------
        val : Any
            A Python scalar or numpy scalar.

        Returns
        -------
        scalar
            A chopped Python scalar (via `.item()` where applicable).

        Raises
        ------
        ValueError
            If `pychop.get_backend()` is unknown.
        """
        b = pychop.get_backend()

        if b == "numpy":
            return self.chopper(np.asarray(val)).item()

        if b == "torch":
            import torch
            t = torch.as_tensor(val)
            out = self.chopper(t)
            return out.item() if hasattr(out, "item") else out

        if b == "jax":
            import jax.numpy as jnp
            x = jnp.asarray(val)
            out = self.chopper(x)
            return out.item() if hasattr(out, "item") else out

        raise ValueError(f"Unsupported backend for CPFloat: {b!r}")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _coerce_other(self, other: Any):
        """
        Coerce a binary operand to a Python scalar.

        Parameters
        ----------
        other : Any
            Right-hand side operand.

        Returns
        -------
        scalar
            Python scalar compatible with arithmetic.

        Raises
        ------
        ValueError
            If `other` is a CPFloat with a different chopper.
        """
        if isinstance(other, CPFloat):
            if other.chopper != self.chopper:
                raise ValueError("All CPFloat inputs must use the same chopper.")
            return other.value
        return float(other)

    # ---------------------------------------------------------------------
    # Binary arithmetic
    # ---------------------------------------------------------------------
    def __add__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value + self._coerce_other(other), self.chopper)

    def __radd__(self, other: Any) -> "CPFloat":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value - self._coerce_other(other), self.chopper)

    def __rsub__(self, other: Any) -> "CPFloat":
        return CPFloat(self._coerce_other(other) - self.value, self.chopper)

    def __mul__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value * self._coerce_other(other), self.chopper)

    def __rmul__(self, other: Any) -> "CPFloat":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value / self._coerce_other(other), self.chopper)

    def __rtruediv__(self, other: Any) -> "CPFloat":
        return CPFloat(self._coerce_other(other) / self.value, self.chopper)

    def __floordiv__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value // self._coerce_other(other), self.chopper)

    def __rfloordiv__(self, other: Any) -> "CPFloat":
        return CPFloat(self._coerce_other(other) // self.value, self.chopper)

    def __pow__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value ** self._coerce_other(other), self.chopper)

    def __rpow__(self, other: Any) -> "CPFloat":
        return CPFloat(self._coerce_other(other) ** self.value, self.chopper)

    def __mod__(self, other: Any) -> "CPFloat":
        return CPFloat(self.value % self._coerce_other(other), self.chopper)

    def __rmod__(self, other: Any) -> "CPFloat":
        return CPFloat(self._coerce_other(other) % self.value, self.chopper)

    # ---------------------------------------------------------------------
    # Unary arithmetic
    # ---------------------------------------------------------------------
    def __neg__(self) -> "CPFloat":
        return CPFloat(-self.value, self.chopper)

    def __pos__(self) -> "CPFloat":
        return self

    def __abs__(self) -> "CPFloat":
        return CPFloat(abs(self.value), self.chopper)

    # ---------------------------------------------------------------------
    # Comparisons
    # ---------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        return self.value == self._coerce_other(other)

    def __ne__(self, other: Any) -> bool:
        return self.value != self._coerce_other(other)

    def __lt__(self, other: Any) -> bool:
        return self.value < self._coerce_other(other)

    def __le__(self, other: Any) -> bool:
        return self.value <= self._coerce_other(other)

    def __gt__(self, other: Any) -> bool:
        return self.value > self._coerce_other(other)

    def __ge__(self, other: Any) -> bool:
        return self.value >= self._coerce_other(other)

    # ---------------------------------------------------------------------
    # Conversions / NumPy interop
    # ---------------------------------------------------------------------
    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(float(self.value))

    def item(self):
        """
        Return the chopped value as a Python scalar.

        Returns
        -------
        scalar
            The stored chopped scalar.
        """
        return self.value

    def to_numpy(self, dtype=None) -> np.ndarray:
        """
        Convert to a NumPy 0-d array.

        Parameters
        ----------
        dtype : numpy dtype, default=None
            Optional dtype to cast to.

        Returns
        -------
        numpy.ndarray
            A 0-d NumPy array containing the chopped value.
        """
        arr = np.asarray(self.value)
        return arr.astype(dtype) if dtype is not None else arr

    __array_priority__ = 1000

    def __array__(self, dtype=None) -> np.ndarray:
        return self.to_numpy(dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        NumPy ufunc protocol.

        This enables expressions such as ``np.sin(CPFloat(...))`` to return a
        ``CPFloat``. Scalar outputs are returned as ``CPFloat``; non-scalar
        outputs are returned as-is.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc being applied.
        method : str
            Ufunc method name.
        *inputs : tuple
            Inputs to the ufunc.
        **kwargs : dict
            Keyword arguments passed to the ufunc.

        Returns
        -------
        CPFloat or tuple or numpy.ndarray or scalar
            Wrapped scalar outputs are returned as ``CPFloat``.
        """
        ch = self.chopper
        for x in inputs:
            if isinstance(x, CPFloat) and x.chopper != ch:
                raise ValueError("All CPFloat inputs must use the same chopper.")

        unwrapped = [x.value if isinstance(x, CPFloat) else x for x in inputs]
        out = getattr(ufunc, method)(*unwrapped, **kwargs)

        def wrap_scalar(s):
            if np.isscalar(s) and not isinstance(s, (str, bytes)):
                a = np.asarray(s)
                if a.dtype.kind in "biufc":
                    return CPFloat(a.item(), ch)
            return s

        if isinstance(out, tuple):
            return tuple(wrap_scalar(v) for v in out)
        if np.isscalar(out):
            return wrap_scalar(out)
        return out

    # ---------------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------------
    def __str__(self) -> str:
        prec_info = (
            f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}"
            if hasattr(self.chopper, "exp_bits")
            else "custom"
        )
        return f"CPFloat({self.value}, {prec_info})"

    def __repr__(self) -> str:
        return str(self)