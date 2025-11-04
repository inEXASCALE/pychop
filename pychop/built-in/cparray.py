import numpy as np
from pychop import LightChop  # Or: from pychop import Chop

class ChoppedArray(np.ndarray):
    """
    A NumPy array subclass that maintains chopped precision after arithmetic ops.
    - Inherits from np.ndarray for full compatibility.
    - Uses LightChop for rounding arrays.
    - Operations return ChoppedArray instances (chopped post-op).
    Fixed: Avoids recursion by chopping pure ndarrays only.
    """
    def __new__(cls, input_array, chopper=None):
        if chopper is None:
            raise ValueError("Must provide a chopper (LightChop or Chop instance)")
        # Chop the base array FIRST (pure ndarray) to avoid subclass recursion
        base_input = np.asarray(input_array)  # Strip any subclass
        chopped_base = chopper(base_input)    # LightChop on pure -> pure chopped ndarray
        # Now view the pre-chopped base as ChoppedArray (no re-chop)
        obj = chopped_base.view(cls)
        obj.chopper = chopper
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.chopper = getattr(obj, 'chopper', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override for ufuncs (+, -, *, /, etc.): Compute on pure arrays, chop pure result, view as ChoppedArray.
        """
        # Validate same chopper for ChoppedArray inputs
        for inp in inputs:
            if isinstance(inp, ChoppedArray) and inp.chopper != self.chopper:
                raise ValueError("All ChoppedArray inputs must use the same chopper")

        # Compute on pure ndarrays
        full_inputs = [np.asarray(x) for x in inputs]  # Strip subclasses
        result = getattr(ufunc, method)(*full_inputs, **kwargs)  # Pure computation

        # Chop the pure result
        chopped_result = self.chopper(result)  # LightChop on pure -> pure chopped

        # Return as ChoppedArray (views pre-chopped; no recursion)
        if chopped_result.ndim == 0:
            return chopped_result.item()  # Scalar fallback
        else:
            return ChoppedArray(chopped_result, self.chopper)  # Safe view

    # Matmul: Strip self to pure before computation
    def __matmul__(self, other):
        self_pure = self.view(np.ndarray)  # Strip subclass
        other_pure = np.asarray(other)
        result = np.matmul(self_pure, other_pure)
        return ChoppedArray(result, self.chopper)  # Views pre-chopped result

    def __rmatmul__(self, other):
        return ChoppedArray(np.matmul(np.asarray(other), self.view(np.ndarray)), self.chopper)

    # Utility: View as regular array
    def to_regular(self):
        return np.asarray(self)

    def __str__(self):
        prec_info = f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}" if hasattr(self.chopper, 'exp_bits') else "custom"
        return f"ChoppedArray({np.array2string(self)}, {prec_info})"

    def __repr__(self):
        return str(self)
