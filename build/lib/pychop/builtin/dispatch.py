"""
pychop.builtin.dispatch
======================

A small utility layer for *unwrapping* chopped-precision container types
(CPArray / CPJaxArray / CPTensor / CPFloat-like) into native backend arrays,
calling a backend function, and then *chopping + wrapping* outputs back into
the chopped-precision types.

This is intended for "high-level algorithms" (e.g., linalg routines) where the
backend library (NumPy/SciPy/JAX/Torch) expects native arrays/tensors.

Design
------
- Find a common `chopper` from the first chopped-precision argument.
- Validate all chopped-precision arguments use the same `chopper`.
- Unwrap CP* inputs to native arrays/tensors.
- Call the backend function.
- Wrap numeric array/tensor outputs back to CP* and chop them.
- For scalar outputs, choose a strategy via `scalar_mode`:
    * "python": return a chopped Python scalar (float/int/complex)
    * "cpfloat": return a CPFloat instance (recommended for scalar chaining)

Notes
-----
- This layer guarantees "chop at the function boundary", not inside the
  backend algorithm's internal steps (BLAS/LAPACK/XLA kernels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ChopWrapSpec:
    """
    Specification for one chopped-precision wrapper type.

    Parameters
    ----------
    unwrap : callable
        Function that maps CP* -> native array/tensor.
    wrap : callable
        Function that maps native output + chopper -> CP* (should chop in ctor).
        For non-array outputs, it should typically return the value unchanged.
    is_cp : callable
        Predicate that returns True if the object is an instance of this CP* type.
    """
    unwrap: Callable[[Any], Any]
    wrap: Callable[[Any, Any], Any]
    is_cp: Callable[[Any], bool]


# -----------------------------------------------------------------------------
# Tree helpers
# -----------------------------------------------------------------------------

def _tree_map(x: Any, f: Callable[[Any], Any]) -> Any:
    if isinstance(x, tuple):
        return tuple(_tree_map(v, f) for v in x)
    if isinstance(x, list):
        return [_tree_map(v, f) for v in x]
    if isinstance(x, dict):
        return {k: _tree_map(v, f) for k, v in x.items()}
    return f(x)


def _tree_walk(x: Any) -> Iterable[Any]:
    if isinstance(x, (tuple, list)):
        for v in x:
            yield from _tree_walk(v)
        return
    if isinstance(x, dict):
        for v in x.values():
            yield from _tree_walk(v)
        return
    yield x


# -----------------------------------------------------------------------------
# Scalar classification
# -----------------------------------------------------------------------------

def _is_string_scalar(x: Any) -> bool:
    return isinstance(x, (str, bytes))


def _is_numpy_numeric_ndarray(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype.kind in "biufc"


def _is_numpy_numeric_scalar(x: Any) -> bool:
    """
    True for numeric Python scalars and NumPy scalar types; False for strings/objects.
    """
    if _is_string_scalar(x):
        return False
    if np.isscalar(x):
        arr = np.asarray(x)
        return arr.dtype.kind in "biufc"
    return False


def _try_extract_python_scalar(x: Any):
    """
    Best-effort: if x is a 0-d array/tensor-like with `.item()`, extract a Python scalar.
    """
    try:
        # jax.Array / torch.Tensor / numpy scalar arrays
        if hasattr(x, "ndim") and getattr(x, "ndim") == 0 and hasattr(x, "item"):
            return x.item()
    except Exception:
        return None
    return None


def _wrap_scalar(y: Any, chopper: Any, scalar_mode: str):
    """
    Wrap numeric scalars according to scalar_mode.

    scalar_mode:
    - 'python': return chopped python scalar
    - 'cpfloat': return CPFloat(chopped)
    """
    if scalar_mode not in ("python", "cpfloat"):
        raise ValueError("scalar_mode must be one of {'python','cpfloat'}")

    # Extract a python scalar from 0-d tensors/arrays (jax/torch) if possible
    y_item = _try_extract_python_scalar(y)
    if y_item is not None:
        y = y_item

    if not _is_numpy_numeric_scalar(y):
        return y

    if scalar_mode == "python":
        return chopper(np.asarray(y)).item()

    # scalar_mode == 'cpfloat'
    from .cpfloat import CPFloat
    return CPFloat(y, chopper)


# -----------------------------------------------------------------------------
# Finding chopper + validation
# -----------------------------------------------------------------------------

def find_first_cp_and_chopper(args: Any, kwargs: Any, specs: List[ChopWrapSpec]) -> Tuple[Any, Any, ChopWrapSpec]:
    """
    Find the first CP* object in args/kwargs and return (obj, obj.chopper, spec).
    """
    for obj in _tree_walk(args):
        for spec in specs:
            if spec.is_cp(obj):
                return obj, getattr(obj, "chopper"), spec
    for obj in _tree_walk(kwargs):
        for spec in specs:
            if spec.is_cp(obj):
                return obj, getattr(obj, "chopper"), spec
    raise ValueError("No chopped-precision object found in args/kwargs; cannot determine chopper.")


def validate_same_chopper(args: Any, kwargs: Any, specs: List[ChopWrapSpec], chopper: Any) -> None:
    """
    Ensure all CP* objects in args/kwargs share the same chopper.
    """
    for obj in _tree_walk(args):
        for spec in specs:
            if spec.is_cp(obj) and getattr(obj, "chopper") != chopper:
                raise ValueError("All chopped-precision inputs must use the same chopper.")
    for obj in _tree_walk(kwargs):
        for spec in specs:
            if spec.is_cp(obj) and getattr(obj, "chopper") != chopper:
                raise ValueError("All chopped-precision inputs must use the same chopper.")


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

def chopwrap_call(
    func: Callable[..., Any],
    *args: Any,
    specs: List[ChopWrapSpec],
    scalar_mode: str = "python",
    wrap_numpy_arrays: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Call `func(*args, **kwargs)` by unwrapping CP* inputs and wrapping outputs.

    Parameters
    ----------
    func : callable
        The backend function to call (e.g., `np.linalg.svd`, `torch.linalg.eigh`).
    *args, **kwargs
        Arguments passed to `func`. They may include CP* objects.
    specs : list of ChopWrapSpec
        Specifications for the CP* types to unwrap/wrap.
    scalar_mode : {'python', 'cpfloat'}, default='python'
        Strategy for numeric scalar outputs:
        - 'python': return a chopped python scalar
        - 'cpfloat': return a CPFloat
        This applies to:
          * Python/NumPy scalar outputs
          * 0-d torch/jax outputs (via `.item()` extraction)
    wrap_numpy_arrays : bool, default=True
        If True, numeric NumPy ndarray outputs are chopped+wrapped.
        If False, NumPy arrays are returned as-is (rarely desired).

    Returns
    -------
    Any
        The function output with numeric arrays/tensors chopped+wrapped into CP*
        and numeric scalars returned according to `scalar_mode`.

    Raises
    ------
    ValueError
        If no CP* is present in the input (cannot determine chopper), or if
        multiple CP* inputs use different choppers.
    """
    if kwargs is None:
        kwargs = {}

    _, chopper, primary_spec = find_first_cp_and_chopper(args, kwargs, specs)
    validate_same_chopper(args, kwargs, specs, chopper)

    def unwrap_one(x: Any) -> Any:
        for spec in specs:
            if spec.is_cp(x):
                return spec.unwrap(x)
        return x

    native_args = _tree_map(args, unwrap_one)
    native_kwargs = _tree_map(kwargs, unwrap_one)

    out = func(*native_args, **native_kwargs)

    def wrap_one(y: Any) -> Any:
        # 1) NumPy arrays: wrap numeric arrays only
        if isinstance(y, np.ndarray):
            if wrap_numpy_arrays and _is_numpy_numeric_ndarray(y):
                return primary_spec.wrap(y, chopper)
            return y

        # 2) Scalars (including 0-d torch/jax): wrap according to scalar_mode
        #    This catches python/np scalars and 0-d array/tensor-like outputs.
        y2 = _wrap_scalar(y, chopper, scalar_mode=scalar_mode)
        if y2 is not y:
            return y2

        # 3) Non-NumPy array-like outputs (torch/jax): let spec.wrap decide
        #    Only attempt if it looks array-like.
        if hasattr(y, "shape") and hasattr(y, "dtype"):
            try:
                return primary_spec.wrap(y, chopper)
            except Exception:
                return y

        # 4) Anything else: passthrough
        return y

    return _tree_map(out, wrap_one)