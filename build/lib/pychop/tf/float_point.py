import tensorflow as tf

from ..np.float_point import Chop_ as NPChop
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


class Chop_:
    def __init__(self, prec='h', subnormal=None, rmode=1, flip=False, explim=1,
                 p=0.5, randfunc=None, customs=None, random_state=0):
        self.prec = prec
        self.subnormal = subnormal
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        self.randfunc = randfunc
        self.customs = customs
        self.random_state = random_state
        self._np_impl = NPChop(
            prec=prec,
            subnormal=subnormal,
            rmode=rmode,
            flip=flip,
            explim=explim,
            p=p,
            randfunc=randfunc,
            customs=customs,
            random_state=random_state,
        )
        self.u = getattr(self._np_impl, 'u', None)

    def chop_wrapper(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)
        return unary_numpy_op(
            x,
            lambda arr: self._np_impl(arr),
            tout=x.dtype,
            identity_grad=x.dtype.is_floating,
        )

    def __call__(self, x):
        return self.chop_wrapper(x)

    def options(self):
        return self._np_impl.options()


def _make_unary(name):
    def method(self, x, *args, **kwargs):
        x = tf.convert_to_tensor(x)
        fn = getattr(self._np_impl, name)
        return unary_numpy_op(
            x,
            lambda arr: fn(arr, *args, **kwargs),
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
            lambda arr_x, arr_y: fn(arr_x, arr_y, *args, **kwargs),
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
            lambda arr: fn(arr, axis=axis),
            tout=x.dtype if x.dtype.is_floating else tf.float32,
            identity_grad=x.dtype.is_floating,
            shape_like=tf.reduce_sum(x, axis=axis),
        )
    return method


for _name in _UNARY_METHODS:
    setattr(Chop_, _name, _make_unary(_name))

for _name in _BINARY_METHODS:
    setattr(Chop_, _name, _make_binary(_name))

for _name in _REDUCTION_METHODS:
    setattr(Chop_, _name, _make_reduction(_name))
