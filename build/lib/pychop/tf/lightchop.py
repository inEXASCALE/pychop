import threading
import numpy as np
import tensorflow as tf

from ..np.lightchop import LightChop_ as NPLightChop
from .common import unary_numpy_op, binary_numpy_op


_UNARY_METHODS = [
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
    'arcsinh', 'arccosh', 'arctanh', 'exp', 'expm1', 'log', 'log10', 'log2',
    'log1p', 'sqrt', 'cbrt', 'abs', 'reciprocal', 'square', 'angle', 'real',
    'imag', 'conj', 'floor', 'ceil', 'sign', 'erf', 'erfc', 'gamma', 'fabs',
    'degrees', 'radians', 'cumsum', 'cumprod'
]
_BINARY_METHODS = [
    'hypot', 'power', 'maximum', 'minimum', 'multiply', 'mod', 'divide', 'add',
    'subtract', 'floor_divide', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
    'dot', 'matmul', 'logaddexp'
]
_REDUCTION_METHODS = ['sum', 'prod', 'mean', 'std', 'var']


class LightChop_:
    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, chunk_size=800, random_state=42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.subnormal = subnormal
        self.chunk_size = chunk_size
        self.random_state = random_state

        # Per-instance RNG state for stochastic rounding modes.
        # The NumPy implementation is called through tf.numpy_function, so this
        # wrapper protects NumPy's global RNG state and makes rmode=5/6 use an
        # instance-local deterministic random stream without changing the public API.
        self._np_random_state = np.random.RandomState(random_state)
        self._rng_lock = threading.RLock()

        self._np_impl = NPLightChop(
            exp_bits=exp_bits,
            sig_bits=sig_bits,
            rmode=rmode,
            subnormal=subnormal,
            chunk_size=chunk_size,
            random_state=random_state,
        )
        self.u = getattr(self._np_impl, 'u', None)

    def _call_np_impl(self, fn, *args, **kwargs):
        if self.rmode not in (5, 6):
            return fn(*args, **kwargs)

        with self._rng_lock:
            global_state = np.random.get_state()
            try:
                np.random.set_state(self._np_random_state.get_state())
                out = fn(*args, **kwargs)
                self._np_random_state.set_state(np.random.get_state())
                return out
            finally:
                np.random.set_state(global_state)

    def quantize(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)
        return unary_numpy_op(
            x,
            lambda arr: self._call_np_impl(self._np_impl.quantize, arr),
            tout=x.dtype,
            identity_grad=x.dtype.is_floating,
        )

    def diff(self, x, n=1):
        x = tf.convert_to_tensor(x)
        out = tf.numpy_function(
            lambda arr: self._call_np_impl(self._np_impl.diff, arr, n=n),
            [x],
            Tout=x.dtype,
        )
        out.set_shape(None)
        return out

    def frexp(self, x):
        x = tf.convert_to_tensor(x)
        mantissa, exponent = tf.numpy_function(
            lambda arr: self._call_np_impl(self._np_impl.frexp, arr),
            [x],
            Tout=[x.dtype if x.dtype.is_floating else tf.float32, tf.int32],
        )
        mantissa.set_shape(x.shape)
        exponent.set_shape(x.shape)
        return mantissa, exponent

    def modf(self, x):
        x = tf.convert_to_tensor(x)
        frac, integ = tf.numpy_function(
            lambda arr: self._call_np_impl(self._np_impl.modf, arr),
            [x],
            Tout=[
                x.dtype if x.dtype.is_floating else tf.float32,
                x.dtype if x.dtype.is_floating else tf.float32,
            ],
        )
        frac.set_shape(x.shape)
        integ.set_shape(x.shape)
        return frac, integ

    def ldexp(self, x, i):
        x = tf.convert_to_tensor(x)
        i = tf.convert_to_tensor(i)
        return binary_numpy_op(
            x,
            i,
            lambda arr_x, arr_i: self._call_np_impl(self._np_impl.ldexp, arr_x, arr_i),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            grad_x=x.dtype.is_floating,
            grad_y=False,
            shape_like=x,
        )

    def round(self, x, decimals=0):
        x = tf.convert_to_tensor(x)
        return unary_numpy_op(
            x,
            lambda arr: self._call_np_impl(self._np_impl.round, arr, decimals=decimals),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            identity_grad=x.dtype.is_floating,
        )

    def clip(self, x, a_min, a_max):
        x = tf.convert_to_tensor(x)
        return unary_numpy_op(
            x,
            lambda arr: self._call_np_impl(self._np_impl.clip, arr, a_min=a_min, a_max=a_max),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            identity_grad=x.dtype.is_floating,
        )

    def __call__(self, x):
        return self.quantize(x)


def _make_unary(name):
    def method(self, x, *args, **kwargs):
        x = tf.convert_to_tensor(x)
        fn = getattr(self._np_impl, name)
        return unary_numpy_op(
            x,
            lambda arr: self._call_np_impl(fn, arr, *args, **kwargs),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            identity_grad=x.dtype.is_floating,
        )
    return method


def _make_binary(name):
    def method(self, x, y, *args, **kwargs):
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        fn = getattr(self._np_impl, name)
        return binary_numpy_op(
            x,
            y,
            lambda arr_x, arr_y: self._call_np_impl(fn, arr_x, arr_y, *args, **kwargs),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            grad_x=x.dtype.is_floating,
            grad_y=y.dtype.is_floating,
            shape_like=x,
        )
    return method


def _make_reduction(name):
    def method(self, x, axis=None):
        x = tf.convert_to_tensor(x)
        fn = getattr(self._np_impl, name)
        return unary_numpy_op(
            x,
            lambda arr: self._call_np_impl(fn, arr, axis=axis),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            identity_grad=x.dtype.is_floating,
            shape_like=tf.reduce_sum(x, axis=axis),
        )
    return method


for _name in _UNARY_METHODS:
    setattr(LightChop_, _name, _make_unary(_name))

for _name in _BINARY_METHODS:
    setattr(LightChop_, _name, _make_binary(_name))

for _name in _REDUCTION_METHODS:
    setattr(LightChop_, _name, _make_reduction(_name))
