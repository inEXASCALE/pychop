"""Native TensorFlow operations; no numpy_function or host computation."""
import tensorflow as tf


@tf.custom_gradient
def _ste(x, quantized):
    def gradient(dy):
        return tf.cast(dy, x.dtype), tf.zeros_like(quantized)
    return quantized, gradient


class TensorFlowOps:
    name = "tensorflow"
    int_dtype = tf.int64
    float32 = tf.float32
    float64 = tf.float64

    def values(self, x):
        if not (x.dtype.is_floating or x.dtype.is_integer):
            raise TypeError("P3109 requires real floating-point or integer tensors")
        dtype = x.dtype if x.dtype in (tf.float32, tf.float64) else tf.float32
        return tf.stop_gradient(tf.cast(x, dtype))

    def asarray(self, x, like):
        return tf.convert_to_tensor(x)

    def cast(self, x, dtype):
        return tf.cast(x, dtype)

    def bitcast(self, x):
        return tf.bitcast(x, tf.int64 if x.dtype == tf.float64 else tf.int32)

    def where(self, condition, a, b):
        # TensorFlow does not implicitly promote tensor operands like NumPy.
        if tf.is_tensor(a):
            b = tf.cast(b, a.dtype)
        elif tf.is_tensor(b):
            a = tf.cast(a, b.dtype)
        return tf.where(condition, a, b)

    def maximum(self, x, value):
        return tf.maximum(x, tf.cast(value, x.dtype))

    def power2(self, exponent, like):
        # Integer IEEE exponent fields produce exact powers of two, with explicit
        # underflow to zero. Avoid a log2 approximation at binade boundaries.
        t, bias, max_exp = (52, 1023, 2046) if like.dtype == tf.float64 else (23, 127, 254)
        e = tf.clip_by_value(tf.cast(exponent, self.int_dtype) + bias, 0, max_exp)
        bits = tf.bitwise.left_shift(e, t)
        if like.dtype == tf.float32:
            bits = tf.cast(bits, tf.int32)
        return tf.bitcast(bits, like.dtype)

    floor = staticmethod(tf.floor)
    ceil = staticmethod(tf.math.ceil)
    round = staticmethod(tf.math.rint)
    isnan = staticmethod(tf.math.is_nan)
    isinf = staticmethod(tf.math.is_inf)
    isfinite = staticmethod(tf.math.is_finite)
    broadcast = staticmethod(tf.broadcast_to)

    def integer(self, x):
        return x.dtype.is_integer

    def check(self, valid, message):
        tf.debugging.assert_equal(tf.reduce_all(valid), True, message=message)

    stop = staticmethod(tf.stop_gradient)

    def ste(self, x, q):
        if not x.dtype.is_floating:
            raise TypeError("STE requires floating-point inputs")
        return _ste(x, q)

    bit_and = staticmethod(tf.bitwise.bitwise_and)
    left = staticmethod(tf.bitwise.left_shift)
    right = staticmethod(tf.bitwise.right_shift)

    def decode_dtype(self):
        return self.float64

    shape = staticmethod(tf.shape)
