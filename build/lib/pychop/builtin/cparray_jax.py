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
    - jit:  Supported via jax.pure_callback (chop runs on host, not XLA-compiled).
    - vmap: Supported via jax.pure_callback with vectorized=True.
    - grad: Chop rounding is non-differentiable; use straight-through
            estimators or custom_vjp if gradients are needed.
    """

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
        - If data is a JAX Tracer (inside jit/vmap/scan): use pure_callback
          to run chop on host. This avoids Chop mutating self.rng_key with
          a tracer, which would permanently corrupt Chop's internal state.
        - Otherwise: call chopper directly (fastest path).
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
        """
        Internal: wrap pre-chopped data without re-chopping.
        Used by pytree unflatten and slicing.
        """
        obj = object.__new__(cls)
        obj._data = data
        obj.chopper = chopper
        return obj

    @classmethod
    def _from_result(cls, data, chopper):
        """Chop a computation result and wrap as CPJaxArray."""
        chopped = cls._safe_chop(chopper, data)
        return cls._wrap(chopped, chopper)

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
            return result.item()
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
