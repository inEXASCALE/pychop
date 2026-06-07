import tensorflow as tf
from typing import Union, Tuple, Optional

from ..bfp_formats import BFPSpec, BFP_FORMATS, create_bfp_spec
from ..np.bfp_formats import BFPTensor_ as NPBFPTensor
from .common import unary_numpy_op


class BFPTensor_:
    def __init__(self, data, format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8'):
        if isinstance(format, str):
            self.spec = BFP_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            self.spec = create_bfp_spec(*format)
        else:
            self.spec = format
        self._original_format = format
        self.dtype = tf.convert_to_tensor(data).dtype
        self.shape = tf.convert_to_tensor(data).shape
        self._np_impl = NPBFPTensor(tf.convert_to_tensor(data).numpy(), format=self._original_format)

    def dequantize(self):
        out = tf.convert_to_tensor(self._np_impl.dequantize())
        out = tf.cast(out, self.dtype)
        out.set_shape(self.shape)
        return out

    def statistics(self):
        return self._np_impl.statistics()


class BFPQuantizerSTE(tf.keras.layers.Layer):
    def __init__(self, format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8', **kwargs):
        super().__init__(**kwargs)
        self._original_format = format
        if isinstance(format, str):
            self.spec = BFP_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            self.spec = create_bfp_spec(*format)
        else:
            self.spec = format

    def call(self, x):
        x = tf.convert_to_tensor(x)
        fmt = self._original_format
        return unary_numpy_op(
            x,
            lambda arr: NPBFPTensor(arr, format=fmt).dequantize(),
            tout=x.dtype,
            identity_grad=x.dtype.is_floating,
        )


def bfp_quantize(data, format: Union[str, BFPSpec, Tuple[int, int]] = 'bfp8', backend: Optional[str] = None):
    return BFPQuantizerSTE(format=format)(data)