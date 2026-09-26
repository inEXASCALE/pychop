"""Small shared validators for the standalone numerical APIs."""
from numbers import Integral
import numpy as np


def bounded_integer(value, name, lower, upper):
    """Accept Python/NumPy integers, rejecting booleans and silent truncation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return int(value)
