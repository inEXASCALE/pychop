import numpy as np


class CPArray(np.ndarray):
    """
    A NumPy array subclass that maintains chopped precision after arithmetic ops.

    Key behaviors:
    - Construction chops the input immediately.
    - NumPy ufuncs (+, -, *, /, etc.) are intercepted via __array_ufunc__:
        compute on base ndarrays -> chop result -> wrap as CPArray.
    - Matrix multiplication (@) is intercepted via __matmul__/__rmatmul__:
        compute with np.matmul on base ndarrays -> wrap (constructor chops).
    - NumPy high-level functions (including np.linalg.*) are intercepted via
      __array_function__:
        unwrap CPArray inputs to base ndarrays -> call func -> wrap numeric ndarray
        outputs back to CPArray and chop numeric scalar outputs.

    Important safety notes:
    - __array_function__ MUST be conservative to avoid breaking NumPy internals
      (printing/formatting, string/object dtypes, etc.).
    - We only chop/wrap numeric ndarrays (dtype.kind in "biufc") and numeric scalars.
    """

    def __new__(cls, input_array, chopper=None):
        if chopper is None:
            raise ValueError("Must provide a chopper (Chop or Chop instance)")

        # Chop the base array FIRST (pure ndarray) to avoid subclass recursion
        base_input = np.asarray(input_array)   # strip any subclass
        chopped_base = chopper(base_input)     # chop on pure -> pure chopped ndarray

        # View the pre-chopped base as CPArray (no extra re-chop)
        obj = np.asarray(chopped_base).view(cls)
        obj.chopper = chopper
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.chopper = getattr(obj, "chopper", None)

    # ----- Helpers ---------
    @staticmethod
    def _is_numeric_ndarray(x: np.ndarray) -> bool:
        # bool/int/uint/float/complex
        return isinstance(x, np.ndarray) and (x.dtype.kind in "biufc")

    @staticmethod
    def _is_string_scalar(x) -> bool:
        return isinstance(x, (str, bytes))

    # ----- NumPy ufunc interception --------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept NumPy ufuncs (+, -, *, /, **, comparisons, etc.).
        Strategy:
          - Validate chopper consistency among CPArray inputs.
          - Compute using base ndarrays (np.asarray).
          - Chop numeric results.
          - Wrap numeric array results as CPArray.
        """
        # Validate same chopper for CPArray inputs
        for inp in inputs:
            if isinstance(inp, CPArray) and inp.chopper != self.chopper:
                raise ValueError("All CPArray inputs must use the same chopper")

        # Compute on pure ndarrays
        full_inputs = [np.asarray(x) for x in inputs]
        result = getattr(ufunc, method)(*full_inputs, **kwargs)

        # Some ufuncs return tuples (e.g., modf, frexp, divmod, etc.)
        if isinstance(result, tuple):
            return tuple(self._wrap_ufunc_result(r) for r in result)
        return self._wrap_ufunc_result(result)

    def _wrap_ufunc_result(self, result):
        # For ndarray results: only chop numeric arrays; let others pass through
        if isinstance(result, np.ndarray):
            if self._is_numeric_ndarray(result):
                chopped = self.chopper(result)
                # Preserve scalar fallback as python scalar
                if np.asarray(chopped).ndim == 0:
                    return np.asarray(chopped).item()
                return CPArray(chopped, self.chopper)
            return result

        # For scalar results: only chop numeric scalars
        if np.isscalar(result):
            if self._is_string_scalar(result):
                return result
            r = np.asarray(result)
            if r.dtype.kind in "biufc":
                return self.chopper(r).item()
            return result

        # For anything else: pass through
        return result

    # ----- Matmul interception -------------------------------------
    def __matmul__(self, other):
        self_pure = self.view(np.ndarray)  # strip subclass
        other_pure = np.asarray(other)
        result = np.matmul(self_pure, other_pure)
        return CPArray(result, self.chopper)  # constructor will chop

    def __rmatmul__(self, other):
        result = np.matmul(np.asarray(other), self.view(np.ndarray))
        return CPArray(result, self.chopper)  # constructor will chop

    # ----- NumPy high-level function interception -------------------
    def __array_function__(self, func, types, args, kwargs):
        """
        Intercept NumPy high-level functions (including np.linalg.*).

        Conservative strategy:
          - Only participate when ALL involved array types are ndarrays/CPArray.
          - Unwrap CPArray inputs to base ndarrays.
          - Call the original NumPy function.
          - Wrap only *numeric* ndarray outputs back into CPArray, using the
            common chopper from inputs.
          - Chop only *numeric* scalar outputs.

        This avoids breaking non-numeric pathways (printing, string/object arrays,
        dtype inspection, formatting utilities, etc.).
        """
        if kwargs is None:
            kwargs = {}

        # If other non-ndarray types are involved, do not override dispatch
        if not all(issubclass(t, (np.ndarray, CPArray)) for t in types):
            return NotImplemented

        choppers = []

        def unwrap(x):
            if isinstance(x, CPArray):
                choppers.append(x.chopper)
                return np.asarray(x)
            return x

        def unwrap_tree(x):
            if isinstance(x, (tuple, list)):
                return type(x)(unwrap_tree(v) for v in x)
            if isinstance(x, dict):
                return {k: unwrap_tree(v) for k, v in x.items()}
            return unwrap(x)

        new_args = unwrap_tree(args)
        new_kwargs = unwrap_tree(kwargs)

        # If no CPArray inputs, we don't need to handle it
        if len(choppers) == 0:
            return NotImplemented

        # Ensure chopper consistency
        chopper = choppers[0]
        if any(c != chopper for c in choppers[1:]):
            raise ValueError("All CPArray inputs must use the same chopper")

        out = func(*new_args, **new_kwargs)

        def wrap_out(y):
            # Wrap numeric ndarray outputs only
            if isinstance(y, np.ndarray):
                if self._is_numeric_ndarray(y):
                    return CPArray(y, chopper)
                return y

            # Chop numeric scalar outputs only
            if np.isscalar(y):
                if self._is_string_scalar(y):
                    return y
                yy = np.asarray(y)
                if yy.dtype.kind in "biufc":
                    return chopper(yy).item()
                return y

            return y

        if isinstance(out, tuple):
            return tuple(wrap_out(v) for v in out)
        if isinstance(out, list):
            return [wrap_out(v) for v in out]
        if isinstance(out, dict):
            return {k: wrap_out(v) for k, v in out.items()}
        return wrap_out(out)

    # ----- Utilities ------------------------------------------------
    def to_regular(self):
        """Return as a regular NumPy ndarray (drops CPArray subclass)."""
        return np.asarray(self)

    def __str__(self):
        # Avoid triggering __array_function__ during formatting by converting
        # to base ndarray explicitly.
        prec_info = (
            f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}"
            if hasattr(self.chopper, "exp_bits") else "custom"
        )
        arr = np.asarray(self)
        return f"CPArray({np.array2string(arr)}, {prec_info})"

    def __repr__(self):
        return str(self)