import jax
import jax.numpy as jnp
from pychop import Chop


class CPJaxArray:
    """
    A JAX array wrapper that maintains chopped precision after arithmetic ops.

    What it guarantees:
    - Binary ops implemented here (+, -, *, /, @, etc.) produce CPJaxArray and
      chop the result after each op (via _from_result -> _safe_chop).
    - Can be passed into many JAX APIs because we implement __jax_array__.

    What it does NOT guarantee:
    - JAX library algorithms (jnp.linalg.*, jax.scipy.*) will not "chop every
      internal step" automatically. They will treat CPJaxArray as a JAX array
      (via __jax_array__), run in full precision, and return JAX arrays.
      Use chopwrap(...) below if you want chopped+wrapped outputs.
    """

    __array_priority__ = 1000  # helps in some mixed-operation cases

    def __init__(self, input_array, chopper=None):
        if chopper is None:
            raise ValueError("Must provide a chopper (Chop instance)")
        base_input = jnp.asarray(input_array)
        self._data = self._safe_chop(chopper, base_input)
        self.chopper = chopper

    @staticmethod
    def _safe_chop(chopper, data):
        """
        Apply chopper in a jit-safe way.
        - If data is a JAX Tracer: use pure_callback to run chop on host.
        - Otherwise: call chopper directly.
        """
        if isinstance(data, jax.core.Tracer):
            return jax.pure_callback(
                chopper,
                jax.ShapeDtypeStruct(data.shape, data.dtype),
                data,
            )
        return chopper(data)

    @classmethod
    def _wrap(cls, data, chopper):
        """Wrap pre-chopped data without re-chopping."""
        obj = object.__new__(cls)
        obj._data = data
        obj.chopper = chopper
        return obj

    @classmethod
    def _from_result(cls, data, chopper):
        """Chop a computation result and wrap as CPJaxArray."""
        chopped = cls._safe_chop(chopper, data)
        return cls._wrap(chopped, chopper)

    # ---- JAX interop: allow CPJaxArray to be used as an array input ----
    def __jax_array__(self):
        """
        JAX will call this to coerce CPJaxArray to a jax.Array when needed
        (e.g., inside jnp.linalg.*, jax.scipy.*, jnp.asarray, etc.).
        """
        return self._data

    # ---- Delegate common array attributes to underlying JAX array ----
    def __getattr__(self, name):
        # Called only if attribute not found on CPJaxArray
        return getattr(self._data, name)

    # ── Delegated properties ────────────────────────────────────────
    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    # ── Binary arithmetic helpers ───────────────────────────────────
    def _binop(self, other, op):
        a = self._data
        if isinstance(other, CPJaxArray):
            if other.chopper != self.chopper:
                raise ValueError("All CPJaxArray inputs must use the same chopper")
            b = other._data
        else:
            b = jnp.asarray(other)
        result = op(a, b)
        return self._from_result(result, self.chopper)

    def _rbinop(self, other, op):
        a = jnp.asarray(other)
        b = self._data
        result = op(a, b)
        return self._from_result(result, self.chopper)

    # ── Arithmetic ops ──────────────────────────────────────────────
    def __add__(self, other):       return self._binop(other, jnp.add)
    def __radd__(self, other):      return self._rbinop(other, jnp.add)
    def __sub__(self, other):       return self._binop(other, jnp.subtract)
    def __rsub__(self, other):      return self._rbinop(other, jnp.subtract)
    def __mul__(self, other):       return self._binop(other, jnp.multiply)
    def __rmul__(self, other):      return self._rbinop(other, jnp.multiply)
    def __truediv__(self, other):   return self._binop(other, jnp.divide)
    def __rtruediv__(self, other):  return self._rbinop(other, jnp.divide)
    def __floordiv__(self, other):  return self._binop(other, jnp.floor_divide)
    def __rfloordiv__(self, other): return self._rbinop(other, jnp.floor_divide)
    def __pow__(self, other):       return self._binop(other, jnp.power)
    def __rpow__(self, other):      return self._rbinop(other, jnp.power)
    def __mod__(self, other):       return self._binop(other, jnp.mod)
    def __rmod__(self, other):      return self._rbinop(other, jnp.mod)

    # ── Unary ops ───────────────────────────────────────────────────
    def __neg__(self):
        return self._from_result(-self._data, self.chopper)

    def __abs__(self):
        return self._from_result(jnp.abs(self._data), self.chopper)

    # ── Matrix multiplication ───────────────────────────────────────
    def __matmul__(self, other):
        a = self._data
        if isinstance(other, CPJaxArray):
            if other.chopper != self.chopper:
                raise ValueError("All CPJaxArray inputs must use the same chopper")
            b = other._data
        else:
            b = jnp.asarray(other)
        result = jnp.matmul(a, b)
        return self._from_result(result, self.chopper)

    def __rmatmul__(self, other):
        result = jnp.matmul(jnp.asarray(other), self._data)
        return self._from_result(result, self.chopper)

    # ── Comparison ops (return plain bool arrays) ────────────────────
    def __eq__(self, other):
        b = other._data if isinstance(other, CPJaxArray) else jnp.asarray(other)
        return self._data == b

    def __ne__(self, other):
        b = other._data if isinstance(other, CPJaxArray) else jnp.asarray(other)
        return self._data != b

    def __lt__(self, other):
        b = other._data if isinstance(other, CPJaxArray) else jnp.asarray(other)
        return self._data < b

    def __le__(self, other):
        b = other._data if isinstance(other, CPJaxArray) else jnp.asarray(other)
        return self._data <= b

    def __gt__(self, other):
        b = other._data if isinstance(other, CPJaxArray) else jnp.asarray(other)
        return self._data > b

    def __ge__(self, other):
        b = other._data if isinstance(other, CPJaxArray) else jnp.asarray(other)
        return self._data >= b

    # ── Indexing ────────────────────────────────────────────────────
    def __getitem__(self, key):
        result = self._data[key]
        if getattr(result, "ndim", 0) == 0:
            # JAX scalar -> Python scalar
            return result.item()
        # Slicing is not an arithmetic op; do not re-chop, just wrap
        return CPJaxArray._wrap(result, self.chopper)

    # ── Utility ─────────────────────────────────────────────────────
    def to_regular(self):
        """View as a plain jax.Array."""
        return self._data

    # ── Printing ────────────────────────────────────────────────────
    def __str__(self):
        prec_info = (
            f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}"
            if hasattr(self.chopper, "exp_bits")
            else "custom"
        )
        return f"CPJaxArray({self._data}, {prec_info})"

    def __repr__(self):
        return str(self)


# ---- Helper: run a JAX/JaX-SciPy function then chop+wrap outputs ----
def chopwrap(func, *args, **kwargs):
    """
    Call `func(*args, **kwargs)` where args may contain CPJaxArray, run the
    function on underlying jax arrays, then chop and wrap numeric array outputs
    back into CPJaxArray using the common chopper.

    Typical use:
      w, v = chopwrap(jnp.linalg.eig, A)
      P, L, U = chopwrap(jax.scipy.linalg.lu, A)
    """
    choppers = []

    def unwrap(x):
        if isinstance(x, CPJaxArray):
            choppers.append(x.chopper)
            return x._data
        return x

    def unwrap_tree(x):
        if isinstance(x, (tuple, list)):
            return type(x)(unwrap_tree(v) for v in x)
        if isinstance(x, dict):
            return {k: unwrap_tree(v) for k, v in x.items()}
        return unwrap(x)

    new_args = unwrap_tree(args)
    new_kwargs = unwrap_tree(kwargs)

    if len(choppers) == 0:
        raise ValueError("chopwrap requires at least one CPJaxArray argument")

    chopper = choppers[0]
    if any(c != chopper for c in choppers[1:]):
        raise ValueError("All CPJaxArray inputs must use the same chopper")

    out = func(*new_args, **new_kwargs)

    def wrap_out(y):
        # Wrap JAX arrays with chop+wrap
        # (jax.Array is the common base; jnp.ndarray is an alias-like type)
        if hasattr(y, "shape") and hasattr(y, "dtype") and not isinstance(y, (str, bytes)):
            try:
                y_arr = jnp.asarray(y)
                # Only chop numeric types
                if y_arr.dtype.kind in "biufc":
                    return CPJaxArray._from_result(y_arr, chopper)
            except Exception:
                pass
        # Scalars: only chop numeric scalars
        if isinstance(y, (int, float, complex, np.number)):
            return chopper(jnp.asarray(y)).item()
        return y

    if isinstance(out, tuple):
        return tuple(wrap_out(v) for v in out)
    if isinstance(out, list):
        return [wrap_out(v) for v in out]
    if isinstance(out, dict):
        return {k: wrap_out(v) for k, v in out.items()}
    return wrap_out(out)


# ── JAX pytree registration ────────────────────────────────────────
def _cpjaxarray_flatten(x):
    children = (x._data,)
    aux_data = (x.chopper,)
    return children, aux_data


def _cpjaxarray_unflatten(aux_data, children):
    chopper = aux_data[0]
    return CPJaxArray._wrap(children[0], chopper)


jax.tree_util.register_pytree_node(
    CPJaxArray,
    _cpjaxarray_flatten,
    _cpjaxarray_unflatten,
)