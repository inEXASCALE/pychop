import jax
import jax.numpy as jnp
from pychop import Chop


class CPJaxArray:
    """
    A JAX array wrapper that maintains chopped precision after arithmetic ops.
    - Wraps a jax.Array (JAX arrays cannot be subclassed).
    - Uses Chop for rounding arrays.
    - Operations return CPJaxArray instances (chopped post-op).
    - Registered as a JAX pytree node for jit/grad compatibility.

    Note on JAX transformations:
    - jit:  Works if Chop.__call__ is compatible with JAX tracing.
    - vmap: Requires Chop.__call__ to support batched tracers.
            If Chop uses pure-Python scalar ops internally, vmap will fail.
    - grad: Chop rounding is non-differentiable; use straight-through
            estimators or custom_vjp if gradients are needed.
    """

    def __init__(self, input_array, chopper=None):
        if chopper is None:
            raise ValueError("Must provide a chopper (Chop instance)")
        base_input = jnp.asarray(input_array)           # Strip any wrapper
        self._data = chopper(base_input)                 # Chop on pure -> pure chopped
        self.chopper = chopper

    @classmethod
    def _wrap(cls, data, chopper):
        """
        Internal: wrap pre-chopped data without re-chopping.
        Used by pytree unflatten and internal paths where data is already chopped.
        """
        obj = object.__new__(cls)
        obj._data = data
        obj.chopper = chopper
        return obj

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
        # Must chop the result -> go through __init__
        return CPJaxArray(result, self.chopper)

    def _rbinop(self, other, op):
        a = jnp.asarray(other)
        b = self._data
        result = op(a, b)
        return CPJaxArray(result, self.chopper)

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
        return CPJaxArray(-self._data, self.chopper)

    def __abs__(self):
        return CPJaxArray(jnp.abs(self._data), self.chopper)

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
        return CPJaxArray(result, self.chopper)

    def __rmatmul__(self, other):
        return CPJaxArray(
            jnp.matmul(jnp.asarray(other), self._data), self.chopper
        )

    # ── Comparison ops (return plain bool arrays, no chopping) ──────
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
        if result.ndim == 0:
            return result.item()          # Scalar fallback (same as CPArray)
        return CPJaxArray._wrap(result, self.chopper)  # Slice is already chopped

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


# ── JAX pytree registration ────────────────────────────────────────
# _data  -> dynamic leaf  (traced / differentiated by JAX)
# chopper -> static aux   (not traced; changes trigger recompilation)

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
