"""
Backend-aware low-precision math simulation utilities.

This module provides mathematical functions that simulate
low-precision arithmetic using a provided ``chop`` object.


Backend is inferred from input array type (NumPy, PyTorch, or JAX).
"""

from typing import Any, Callable

# ---------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------


def _detect_backend(x: Any) -> str:
    """
    Detect backend from input array type.

    Parameters
    ----------
    x : Any
        Input array or scalar.

    Returns
    -------
    str
        Backend name: 'numpy', 'torch', or 'jax'.
    """
    module = type(x).__module__

    if "torch" in module:
        return "torch"
    if "jax" in module:
        return "jax"
    return "numpy"


def _get_backend_module(backend: str):
    """
    Return backend math module.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    module
        Python module for the backend.
    """
    if backend == "torch":
        import torch
        return torch
    elif backend == "jax":
        import jax.numpy as jnp
        return jnp
    else:
        import numpy as np
        return np


# ---------------------------------------------------------------------
# Safe boolean helpers 
# ---------------------------------------------------------------------


def _to_bool_scalar(x):
    """
    Safely convert backend scalar tensor/array to Python bool.

    Works for NumPy, PyTorch, JAX.

    Parameters
    ----------
    x : scalar
        Input scalar.

    Returns
    -------
    bool
        Python boolean.
    """
    if hasattr(x, "item"):
        return bool(x.item())
    return bool(x)


def _all_true(cond):
    """
    Safely evaluate xp.all(...) result.

    Parameters
    ----------
    cond : scalar
        Backend array/scalar result of comparison.

    Returns
    -------
    bool
        True if all elements satisfy condition.
    """
    return _to_bool_scalar(cond)


# ---------------------------------------------------------------------
# Core wrappers
# ---------------------------------------------------------------------


def _unary_math(chop, x, fn_name: str, domain_check: Callable = None):
    """
    Apply unary math operation with chopping.

    Parameters
    ----------
    chop : callable
        Chopping function.
    x : array_like
        Input array/tensor.
    fn_name : str
        Name of math function (e.g., 'sin', 'exp').
    domain_check : callable, optional
        Function to validate domain.

    Returns
    -------
    array_like
        Chopped result.
    """
    backend = _detect_backend(x)
    
    xp = _get_backend_module(backend)

    # Round input
    x = chop(x)

    if domain_check is not None:
        domain_check(xp, x)

    # Working precision computation
    result = getattr(xp, fn_name)(x)

    # Round output
    return chop(result)


def _binary_math(chop, x, y, fn_name: str, domain_check=None):
    """
    Apply binary math operation with chopping.

    Parameters
    ----------
    chop : callable
        Chopping function.
    x, y : array_like
        Input arrays or scalars.
    fn_name : str
        Name of math function (e.g., 'add', 'power', 'matmul').
    domain_check : callable, optional
        Function to validate domain.

    Returns
    -------
    array_like
        Chopped result.

    Raises
    ------
    ValueError
        If `matmul` inputs are scalar or domain check fails.
    """

    backend = _detect_backend(x)
    
    xp = _get_backend_module(backend)

    x = chop(x)
    y = chop(y)

    if domain_check is not None:
        domain_check(xp, x, y)

    # ---- Special handling for matmul ----
    if fn_name == "matmul":
        if getattr(x, "ndim", 0) == 0 or getattr(y, "ndim", 0) == 0:
            raise ValueError(
                "matmul requires at least 1D inputs. "
                f"Got shapes {getattr(x, 'shape', None)} and {getattr(y, 'shape', None)}."
            )

    result = getattr(xp, fn_name)(x, y)

    return chop(result)


def _reduction_math(chop, x, fn_name: str, **kwargs):
    """
    Apply reduction math operation with chopping.

    Parameters
    ----------
    chop : callable
        Chopping function.
    x : array_like
        Input array/tensor.
    fn_name : str
        Name of reduction function (e.g., 'sum', 'mean').
    kwargs : dict
        Additional keyword arguments (axis, etc.)

    Returns
    -------
    array_like
        Chopped result.
    """
    backend = _detect_backend(x)
    
    xp = _get_backend_module(backend)

    x = chop(x)
    result = getattr(xp, fn_name)(x, **kwargs)
    return chop(result)


# ---------------------------------------------------------------------
# Domain checks 
# ---------------------------------------------------------------------


def _check_positive(xp, x):
    """Check that all elements are positive."""
    if not _all_true(xp.all(x > 0)):
        raise ValueError("Input must be positive.")


def _check_nonnegative(xp, x):
    """Check that all elements are non-negative."""
    if not _all_true(xp.all(x >= 0)):
        raise ValueError("Input must be non-negative.")


def _check_abs_le_one(xp, x):
    """Check that all elements are in [-1, 1]."""
    if not _all_true(xp.all(xp.abs(x) <= 1)):
        raise ValueError("Input must be in [-1, 1].")


def _check_abs_lt_one(xp, x):
    """Check that all elements are in (-1, 1)."""
    if not _all_true(xp.all(xp.abs(x) < 1)):
        raise ValueError("Input must be in (-1, 1).")


def _check_nonzero(xp, x, y=None):
    """Check that all elements are non-zero (or divisor y)."""
    target = y if y is not None else x
    if not _all_true(xp.all(target != 0)):
        raise ValueError("Divisor must not be zero.")


# ---------------------------------------------------------------------
# Trigonometric
# ---------------------------------------------------------------------

def sin(x, chop): return _unary_math(chop, x, "sin")
def cos(x, chop): return _unary_math(chop, x, "cos")
def tan(x, chop): return _unary_math(chop, x, "tan")
def arcsin(x, chop): return _unary_math(chop, x, "arcsin", _check_abs_le_one)
def arccos(x, chop): return _unary_math(chop, x, "arccos", _check_abs_le_one)
def arctan(x, chop): return _unary_math(chop, x, "arctan")

# Hyperbolic
def sinh(x, chop): return _unary_math(chop, x, "sinh")
def cosh(x, chop): return _unary_math(chop, x, "cosh")
def tanh(x, chop): return _unary_math(chop, x, "tanh")
def arcsinh(x, chop): return _unary_math(chop, x, "arcsinh")
def arccosh(x, chop): return _unary_math(chop, x, "arccosh", _check_nonnegative)
def arctanh(x, chop): return _unary_math(chop, x, "arctanh", _check_abs_lt_one)

# Exponential / Log
def exp(x, chop): return _unary_math(chop, x, "exp")
def expm1(x, chop): return _unary_math(chop, x, "expm1")
def log(x, chop): return _unary_math(chop, x, "log", _check_positive)
def log10(x, chop): return _unary_math(chop, x, "log10", _check_positive)
def log2(x, chop): return _unary_math(chop, x, "log2", _check_positive)
def log1p(x, chop): return _unary_math(chop, x, "log1p")

# Power / Roots
def sqrt(x, chop): return _unary_math(chop, x, "sqrt", _check_nonnegative)
def square(x, chop): return _unary_math(chop, x, "square")
def power(x, y, chop): return _binary_math(chop, x, y, "power")

# Arithmetic
def add(x, y, chop): return _binary_math(chop, x, y, "add")
def subtract(x, y, chop): return _binary_math(chop, x, y, "subtract")
def multiply(x, y, chop): return _binary_math(chop, x, y, "multiply")
def divide(x, y, chop): return _binary_math(chop, x, y, "divide", _check_nonzero)
def floor_divide(x, y, chop): return _binary_math(chop, x, y, "floor_divide", _check_nonzero)
def mod(x, y, chop): return _binary_math(chop, x, y, "mod", _check_nonzero)

# Linear algebra
def dot(x, y, chop): return _binary_math(chop, x, y, "dot")
def matmul(x, y, chop): return _binary_math(chop, x, y, "matmul")

# Reductions
def sum(x, chop, axis=None): return _reduction_math(chop, x, "sum", axis=axis)
def prod(x, chop, axis=None): return _reduction_math(chop, x, "prod", axis=axis)
def mean(x, chop, axis=None): return _reduction_math(chop, x, "mean", axis=axis)
def std(x, chop, axis=None): return _reduction_math(chop, x, "std", axis=axis)
def var(x, chop, axis=None): return _reduction_math(chop, x, "var", axis=axis)

# Rounding
def floor(x, chop): return _unary_math(chop, x, "floor")
def ceil(x, chop): return _unary_math(chop, x, "ceil")
def round(x, chop): return _unary_math(chop, x, "round")
def sign(x, chop): return _unary_math(chop, x, "sign")

# Cumulative
def cumsum(x, chop, axis=None): return _reduction_math(chop, x, "cumsum", axis=axis)
def cumprod(x, chop, axis=None): return _reduction_math(chop, x, "cumprod", axis=axis)