import tensorflow as tf

from ..np.integer import Chopi_ as NPChopi
from .common import unary_numpy_op


_BITS_TO_DTYPE = {
    8: tf.int8,
    16: tf.int16,
    32: tf.int32,
    64: tf.int64,
}


class Chopi_:
    def __init__(self, bits=8, symmetric=False, per_channel=False, axis=0, verbose=False):
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.axis = axis
        self.verbose = verbose
        self.intType = _BITS_TO_DTYPE.get(bits, tf.int64)
        self._np_impl = NPChopi(
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            axis=axis,
            verbose=verbose,
        )
        self.scale = None
        self.zero_point = None
        self.is_calibrated = False

    def calibrate(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)
        self._np_impl.calibrate(x.numpy())
        self.scale = tf.convert_to_tensor(self._np_impl.scale, dtype=tf.float32)
        self.zero_point = tf.convert_to_tensor(self._np_impl.zero_point)
        self.is_calibrated = True
        return self.scale, self.zero_point

    def quantize(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)
        return unary_numpy_op(
            x,
            lambda arr: self._np_impl.quantize(arr),
            tout=self.intType,
            identity_grad=False,
        )

    def dequantize(self, q):
        q = tf.convert_to_tensor(q)
        return unary_numpy_op(
            q,
            lambda arr: self._np_impl.dequantize(arr),
            tout=tf.float32,
            identity_grad=False,
        )

    def forward(self, x, training=True):
        x = tf.convert_to_tensor(x)
        q = self.quantize(x)
        x_dequant = self.dequantize(q)
        if training and x.dtype.is_floating:
            return x + tf.stop_gradient(tf.cast(x_dequant, x.dtype) - x)
        return tf.cast(x_dequant, x.dtype if x.dtype.is_floating else tf.float32)

    def __call__(self, x):
        return self.forward(x, training=True)
