import numpy as np
from pychop import LightChop  # Use LightChop for faster emulation
# Optional: from pychop import Chop  # If you need full features

class CPFloat:
    """
    A wrapper for scalars that maintains chopped precision after arithmetic ops.
    Uses LightChop (or Chop) for rounding—results stay in the target precision.
    - value: Internal storage (Python float, backed by fp64 for safety).
    - chopper: The LightChop (or Chop) instance for rounding.
    """
    def __init__(self, value, chopper):
        self.chopper = chopper
        self.value = self._chop_scalar(value)

    def _chop_scalar(self, val):
        """Helper: Chop a scalar using the chopper instance."""
        return self.chopper(np.asarray(val)).item()  # Chop and extract scalar

    # Binary arithmetic: Compute in full prec, then chop result
    def __add__(self, other):
        res = self.value + float(other)
        return CPFloat(res, self.chopper)

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):
        res = self.value - float(other)
        return CPFloat(res, self.chopper)

    def __rsub__(self, other):
        return CPFloat(float(other) - self.value, self.chopper)

    def __mul__(self, other):
        res = self.value * float(other)
        return CPFloat(res, self.chopper)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        res = self.value / float(other)
        return CPFloat(res, self.chopper)

    def __rtruediv__(self, other):
        return CPFloat(float(other) / self.value, self.chopper)

    # Unary operations
    def __neg__(self):
        return CPFloat(-self.value, self.chopper)

    def __pos__(self):
        return self

    def __abs__(self):
        return CPFloat(abs(self.value), self.chopper)

    # Comparisons (return bool; no type change)
    def __eq__(self, other):
        return self.value == float(other)

    def __ne__(self, other):
        return self.value != float(other)

    def __lt__(self, other):
        return self.value < float(other)

    def __le__(self, other):
        return self.value <= float(other)

    def __gt__(self, other):
        return self.value > float(other)

    def __ge__(self, other):
        return self.value >= float(other)

    # Conversions
    def __float__(self):
        return self.value

    def __str__(self):
        prec_info = f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}" if hasattr(self.chopper, 'exp_bits') else "custom"
        return f"CPFloat({self.value}, {prec_info})"

    def __repr__(self):
        return str(self)
