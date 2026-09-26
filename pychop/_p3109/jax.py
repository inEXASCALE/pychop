"""Native JAX operations, compatible with jit, vmap and explicit STE gradients."""
import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def _ste(x, quantized):
    return quantized


@_ste.defjvp
def _ste_jvp(primals, tangents):
    return primals[1], tangents[0].astype(primals[1].dtype)


class JaxOps:
    name = "jax"
    float32 = jnp.float32
    float64 = jnp.float64

    @property
    def int_dtype(self):
        return jnp.int64 if jax.config.x64_enabled else jnp.int32

    def values(self, x):
        if not (jnp.issubdtype(x.dtype, jnp.floating) or jnp.issubdtype(x.dtype, jnp.integer)):
            raise TypeError("P3109 requires real floating-point or integer arrays")
        dtype = x.dtype if x.dtype in (jnp.float32, jnp.float64) else jnp.float32
        return jax.lax.stop_gradient(x.astype(dtype))

    def asarray(self, x, like):
        return jnp.asarray(x)

    def cast(self, x, dtype):
        return x.astype(dtype)

    def bitcast(self, x):
        return jax.lax.bitcast_convert_type(x, jnp.int64 if x.dtype == jnp.float64 else jnp.int32)

    where = staticmethod(jnp.where)
    maximum = staticmethod(jnp.maximum)
    floor = staticmethod(jnp.floor)
    ceil = staticmethod(jnp.ceil)
    round = staticmethod(jnp.rint)
    isnan = staticmethod(jnp.isnan)
    isinf = staticmethod(jnp.isinf)
    isfinite = staticmethod(jnp.isfinite)
    broadcast = staticmethod(jnp.broadcast_to)

    def power2(self, exponent, like):
        t, bias, maximum = (52, 1023, 2046) if like.dtype == self.float64 else (23, 127, 254)
        bits = jnp.left_shift(jnp.clip(exponent + bias, 0, maximum), t)
        return jax.lax.bitcast_convert_type(bits.astype(jnp.int64 if t == 52 else jnp.int32), like.dtype)

    def integer(self, x):
        return jnp.issubdtype(x.dtype, jnp.integer)

    def check(self, valid, message):
        if isinstance(valid, jax.core.Tracer):
            raise ValueError("JAX tracing requires check=False for code/random-bit validation; "
                             "validate inputs eagerly before compilation")
        if not np.all(np.asarray(valid)):
            raise ValueError(message)

    stop = staticmethod(jax.lax.stop_gradient)

    def ste(self, x, q):
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError("STE requires floating-point inputs")
        return _ste(x, q)

    bit_and = staticmethod(jnp.bitwise_and)
    left = staticmethod(jnp.left_shift)
    right = staticmethod(jnp.right_shift)

    def decode_dtype(self):
        return self.float64 if jax.config.x64_enabled else self.float32

    shape = staticmethod(lambda x: x.shape)
