import tensorflow as tf
from typing import Union, Tuple, Optional

from ..mx_formats import MXSpec, MX_FORMATS, create_mx_spec
from ..np.mx_formats import MXTensor_ as NPMXTensor
from .common import unary_numpy_op


class MXTensor_:
    def __init__(self, data, format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3', block_size: int = 32,
                 scale_exp_bits: Optional[int] = None, scale_sig_bits: Optional[int] = None):
        if isinstance(format, str):
            self.spec = MX_FORMATS[format.lower()]
        elif isinstance(format, tuple):
            self.spec = create_mx_spec(*format, block_size=block_size)
        else:
            self.spec = format
        self._original_format = format
        self.dtype = tf.convert_to_tensor(data).dtype
        self.shape = tf.convert_to_tensor(data).shape
        self._np_impl = NPMXTensor(
            tf.convert_to_tensor(data).numpy(),
            format=self._original_format,
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            scale_sig_bits=scale_sig_bits,
        )

    def dequantize(self):
        out = tf.convert_to_tensor(self._np_impl.dequantize())
        out = tf.cast(out, self.dtype)
        out.set_shape(self.shape)
        return out

    def statistics(self):
        return self._np_impl.statistics()


class MXQuantizerSTE(tf.keras.layers.Layer):
    def __init__(self, format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3', block_size: int = 32,
                 scale_exp_bits: Optional[int] = None, scale_sig_bits: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.format = format
        self.block_size = block_size
        self.scale_exp_bits = scale_exp_bits
        self.scale_sig_bits = scale_sig_bits

    def call(self, x):
        x = tf.convert_to_tensor(x)
        return unary_numpy_op(
            x,
            lambda arr: NPMXTensor(
                arr,
                format=self.format,
                block_size=self.block_size,
                scale_exp_bits=self.scale_exp_bits,
                scale_sig_bits=self.scale_sig_bits,
            ).dequantize(),
            tout=x.dtype,
            identity_grad=x.dtype.is_floating,
        )


def mx_quantize(data, format: Union[str, MXSpec, Tuple[int, int]] = 'mxfp8_e4m3', block_size: int = 32,
                scale_exp_bits: Optional[int] = None, scale_sig_bits: Optional[int] = None, backend: Optional[str] = None):
    return MXQuantizerSTE(
        format=format,
        block_size=block_size,
        scale_exp_bits=scale_exp_bits,
        scale_sig_bits=scale_sig_bits,
    )(data)