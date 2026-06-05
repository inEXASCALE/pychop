import tensorflow as tf

from ..chop import Chop
from ..fixed_point import Chopf
from ..integer import Chopi
from .ptq import (
    post_quantization,
    static_post_quantization,
    dynamic_post_quantization,
    mixed_post_quantization,
)


class ChopSTE(Chop):
    pass


class ChopfSTE(Chopf):
    pass


class ChopiSTE(Chopi):
    pass


class _QuantizedLayerMixin:
    def __init__(self, *args, chop=None, quantize_input=True, quantize_output=True, quantize_weights=True, **kwargs):
        self.chop = chop
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        self.quantize_weights = quantize_weights
        super().__init__(*args, **kwargs)

    def _apply_chop(self, value):
        if self.chop is None or value is None:
            return value
        return self.chop(value)

    def _prepare_inputs(self, inputs):
        return self._apply_chop(inputs) if self.quantize_input else inputs

    def _finalize_outputs(self, outputs):
        return self._apply_chop(outputs) if self.quantize_output else outputs


class QuantizedLinear(_QuantizedLayerMixin, tf.keras.layers.Dense):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        kernel = self._apply_chop(self.kernel) if self.quantize_weights else self.kernel
        outputs = tf.linalg.matmul(x, kernel)
        if self.use_bias:
            bias = self._apply_chop(self.bias) if self.quantize_weights else self.bias
            outputs = tf.nn.bias_add(outputs, bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return self._finalize_outputs(outputs)


class QuantizedConv1d(_QuantizedLayerMixin, tf.keras.layers.Conv1D):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            kernel = self._apply_chop(self.kernel)
            bias = self._apply_chop(self.bias) if self.use_bias else None
            outputs = tf.nn.conv1d(x, kernel, stride=self.strides[0],
                                   padding=self.padding.upper(),
                                   dilations=self.dilation_rate[0])
            if bias is not None:
                outputs = tf.nn.bias_add(outputs, bias)
            if self.activation is not None:
                outputs = self.activation(outputs)
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedConv2d(_QuantizedLayerMixin, tf.keras.layers.Conv2D):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            kernel = self._apply_chop(self.kernel)
            bias = self._apply_chop(self.bias) if self.use_bias else None
            outputs = tf.nn.conv2d(x, kernel, strides=self.strides,
                                   padding=self.padding.upper(),
                                   dilations=self.dilation_rate)
            if bias is not None:
                outputs = tf.nn.bias_add(outputs, bias)
            if self.activation is not None:
                outputs = self.activation(outputs)
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedConv3d(_QuantizedLayerMixin, tf.keras.layers.Conv3D):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            kernel = self._apply_chop(self.kernel)
            bias = self._apply_chop(self.bias) if self.use_bias else None
            outputs = tf.nn.conv3d(x, kernel, strides=(1,) + tuple(self.strides) + (1,),
                                   padding=self.padding.upper(),
                                   dilations=(1,) + tuple(self.dilation_rate) + (1,))
            if bias is not None:
                outputs = tf.nn.bias_add(outputs, bias)
            if self.activation is not None:
                outputs = self.activation(outputs)
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedConvTranspose1d(_QuantizedLayerMixin, tf.keras.layers.Conv1DTranspose):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            orig_kernel = self.kernel
            self.kernel = self._apply_chop(self.kernel)
            if self.use_bias:
                orig_bias = self.bias
                self.bias = self._apply_chop(self.bias)
            outputs = super().call(x)
            self.kernel = orig_kernel
            if self.use_bias:
                self.bias = orig_bias
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedConvTranspose2d(_QuantizedLayerMixin, tf.keras.layers.Conv2DTranspose):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            orig_kernel = self.kernel
            self.kernel = self._apply_chop(self.kernel)
            if self.use_bias:
                orig_bias = self.bias
                self.bias = self._apply_chop(self.bias)
            outputs = super().call(x)
            self.kernel = orig_kernel
            if self.use_bias:
                self.bias = orig_bias
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedConvTranspose3d(_QuantizedLayerMixin, tf.keras.layers.Conv3DTranspose):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            orig_kernel = self.kernel
            self.kernel = self._apply_chop(self.kernel)
            if self.use_bias:
                orig_bias = self.bias
                self.bias = self._apply_chop(self.bias)
            outputs = super().call(x)
            self.kernel = orig_kernel
            if self.use_bias:
                self.bias = orig_bias
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedRNN(_QuantizedLayerMixin, tf.keras.layers.SimpleRNN):
    def call(self, sequences, initial_state=None, mask=None, training=False):
        x = self._prepare_inputs(sequences)
        if self.chop is not None and self.quantize_weights and self.built:
            cell = self.cell
            orig_kernel = cell.kernel.numpy()
            orig_recurrent = cell.recurrent_kernel.numpy()
            cell.kernel.assign(self._apply_chop(cell.kernel))
            cell.recurrent_kernel.assign(self._apply_chop(cell.recurrent_kernel))
            if cell.use_bias:
                orig_bias = cell.bias.numpy()
                cell.bias.assign(self._apply_chop(cell.bias))
            outputs = super().call(x, initial_state=initial_state, mask=mask, training=training)
            cell.kernel.assign(orig_kernel)
            cell.recurrent_kernel.assign(orig_recurrent)
            if cell.use_bias:
                cell.bias.assign(orig_bias)
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x, initial_state=initial_state, mask=mask, training=training))


class QuantizedLSTM(_QuantizedLayerMixin, tf.keras.layers.LSTM):
    def call(self, sequences, initial_state=None, mask=None, training=False):
        x = self._prepare_inputs(sequences)
        if self.chop is not None and self.quantize_weights and self.built:
            cell = self.cell
            orig_kernel = cell.kernel.numpy()
            orig_recurrent = cell.recurrent_kernel.numpy()
            cell.kernel.assign(self._apply_chop(cell.kernel))
            cell.recurrent_kernel.assign(self._apply_chop(cell.recurrent_kernel))
            if cell.use_bias:
                orig_bias = cell.bias.numpy()
                cell.bias.assign(self._apply_chop(cell.bias))
            outputs = super().call(x, initial_state=initial_state, mask=mask, training=training)
            cell.kernel.assign(orig_kernel)
            cell.recurrent_kernel.assign(orig_recurrent)
            if cell.use_bias:
                cell.bias.assign(orig_bias)
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x, initial_state=initial_state, mask=mask, training=training))


class QuantizedGRU(_QuantizedLayerMixin, tf.keras.layers.GRU):
    def call(self, sequences, initial_state=None, mask=None, training=False):
        x = self._prepare_inputs(sequences)
        if self.chop is not None and self.quantize_weights and self.built:
            cell = self.cell
            orig_kernel = cell.kernel.numpy()
            orig_recurrent = cell.recurrent_kernel.numpy()
            cell.kernel.assign(self._apply_chop(cell.kernel))
            cell.recurrent_kernel.assign(self._apply_chop(cell.recurrent_kernel))
            if cell.use_bias:
                orig_bias = cell.bias.numpy()
                cell.bias.assign(self._apply_chop(cell.bias))
            outputs = super().call(x, initial_state=initial_state, mask=mask, training=training)
            cell.kernel.assign(orig_kernel)
            cell.recurrent_kernel.assign(orig_recurrent)
            if cell.use_bias:
                cell.bias.assign(orig_bias)
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x, initial_state=initial_state, mask=mask, training=training))


class QuantizedMaxPool1d(_QuantizedLayerMixin, tf.keras.layers.MaxPooling1D):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedMaxPool2d(_QuantizedLayerMixin, tf.keras.layers.MaxPooling2D):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedMaxPool3d(_QuantizedLayerMixin, tf.keras.layers.MaxPooling3D):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedAvgPool1d(_QuantizedLayerMixin, tf.keras.layers.AveragePooling1D):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedAvgPool2d(_QuantizedLayerMixin, tf.keras.layers.AveragePooling2D):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedAvgPool3d(_QuantizedLayerMixin, tf.keras.layers.AveragePooling3D):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedAdaptiveAvgPool2d(_QuantizedLayerMixin, tf.keras.layers.Layer):
    def __init__(self, output_size, *args, chop=None, **kwargs):
        super().__init__(*args, chop=chop, **kwargs)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)

    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        x = tf.image.resize(x, self.output_size, method='area')
        return self._finalize_outputs(x)


class QuantizedBatchNorm1d(_QuantizedLayerMixin, tf.keras.layers.BatchNormalization):
    def call(self, inputs, training=False):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built and self.scale:
            orig_gamma = self.gamma
            self.gamma = self._apply_chop(self.gamma)
            if self.center:
                orig_beta = self.beta
                self.beta = self._apply_chop(self.beta)
            outputs = super().call(x, training=training)
            self.gamma = orig_gamma
            if self.center:
                self.beta = orig_beta
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x, training=training))


class QuantizedBatchNorm2d(QuantizedBatchNorm1d):
    pass


class QuantizedBatchNorm3d(QuantizedBatchNorm1d):
    pass


class QuantizedLayerNorm(_QuantizedLayerMixin, tf.keras.layers.LayerNormalization):
    def call(self, inputs):
        x = self._prepare_inputs(inputs)
        if self.chop is not None and self.quantize_weights and self.built:
            if hasattr(self, 'gamma') and self.gamma is not None:
                orig_gamma = self.gamma
                self.gamma = self._apply_chop(self.gamma)
            else:
                orig_gamma = None
            if hasattr(self, 'beta') and self.beta is not None:
                orig_beta = self.beta
                self.beta = self._apply_chop(self.beta)
            else:
                orig_beta = None
            outputs = super().call(x)
            if orig_gamma is not None:
                self.gamma = orig_gamma
            if orig_beta is not None:
                self.beta = orig_beta
            return self._finalize_outputs(outputs)
        return self._finalize_outputs(super().call(x))


class QuantizedInstanceNorm1d(QuantizedLayerNorm):
    pass


class QuantizedInstanceNorm2d(QuantizedLayerNorm):
    pass


class QuantizedInstanceNorm3d(QuantizedLayerNorm):
    pass


class QuantizedGroupNorm(QuantizedLayerNorm):
    pass


class QuantizedMultiheadAttention(_QuantizedLayerMixin, tf.keras.layers.MultiHeadAttention):
    def call(self, query, value=None, key=None, training=False, **kwargs):
        query = self._prepare_inputs(query)
        value = self._prepare_inputs(value if value is not None else query)
        key = self._prepare_inputs(key if key is not None else value)
        if self.chop is not None and self.quantize_weights and self.built:
            # Quantize Q/K/V/output projection kernels and biases using assign/restore
            saved = {}
            for proj_name in ('_query_dense', '_key_dense', '_value_dense', '_output_dense'):
                proj = getattr(self, proj_name, None)
                if proj is None:
                    continue
                if hasattr(proj, 'kernel') and proj.kernel is not None:
                    saved[(proj_name, 'kernel')] = proj.kernel.numpy()
                    proj.kernel.assign(self._apply_chop(proj.kernel))
                if hasattr(proj, 'bias') and proj.bias is not None:
                    saved[(proj_name, 'bias')] = proj.bias.numpy()
                    proj.bias.assign(self._apply_chop(proj.bias))
            outputs = super().call(query=query, value=value, key=key, training=training, **kwargs)
            for (proj_name, attr), orig_val in saved.items():
                getattr(getattr(self, proj_name), attr).assign(orig_val)
            return self._finalize_outputs(outputs)
        outputs = super().call(query=query, value=value, key=key, training=training, **kwargs)
        return self._finalize_outputs(outputs)


class QuantizedReLU(_QuantizedLayerMixin, tf.keras.layers.ReLU):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedSigmoid(_QuantizedLayerMixin, tf.keras.layers.Activation):
    def __init__(self, *args, chop=None, **kwargs):
        super().__init__('sigmoid', *args, chop=chop, **kwargs)

    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedTanh(_QuantizedLayerMixin, tf.keras.layers.Activation):
    def __init__(self, *args, chop=None, **kwargs):
        super().__init__('tanh', *args, chop=chop, **kwargs)

    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedLeakyReLU(_QuantizedLayerMixin, tf.keras.layers.LeakyReLU):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedSoftmax(_QuantizedLayerMixin, tf.keras.layers.Softmax):
    def call(self, inputs, mask=None):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs), mask=mask))


class QuantizedGELU(_QuantizedLayerMixin, tf.keras.layers.Activation):
    def __init__(self, *args, chop=None, **kwargs):
        super().__init__(tf.nn.gelu, *args, chop=chop, **kwargs)

    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedELU(_QuantizedLayerMixin, tf.keras.layers.ELU):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedPReLU(_QuantizedLayerMixin, tf.keras.layers.PReLU):
    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedSiLU(_QuantizedLayerMixin, tf.keras.layers.Activation):
    def __init__(self, *args, chop=None, **kwargs):
        super().__init__(tf.nn.silu, *args, chop=chop, **kwargs)

    def call(self, inputs):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs)))


class QuantizedDropout(_QuantizedLayerMixin, tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return self._finalize_outputs(super().call(self._prepare_inputs(inputs), training=training))


class QuantizedEmbedding(_QuantizedLayerMixin, tf.keras.layers.Embedding):
    def call(self, inputs):
        if self.chop is not None and self.quantize_weights and self.built:
            orig_val = self._embeddings.numpy()
            self._embeddings.assign(self._apply_chop(self._embeddings))
            outputs = super().call(inputs)
            self._embeddings.assign(orig_val)
            return self._finalize_outputs(outputs)
        outputs = super().call(inputs)
        return self._finalize_outputs(outputs)


IQuantizedLinear = QuantizedLinear
IQuantizedConv1d = QuantizedConv1d
IQuantizedConv2d = QuantizedConv2d
IQuantizedConv3d = QuantizedConv3d
IQuantizedConvTranspose1d = QuantizedConvTranspose1d
IQuantizedConvTranspose2d = QuantizedConvTranspose2d
IQuantizedConvTranspose3d = QuantizedConvTranspose3d
IQuantizedRNN = QuantizedRNN
IQuantizedLSTM = QuantizedLSTM
IQuantizedGRU = QuantizedGRU
IQuantizedMaxPool1d = QuantizedMaxPool1d
IQuantizedMaxPool2d = QuantizedMaxPool2d
IQuantizedMaxPool3d = QuantizedMaxPool3d
IQuantizedAvgPool1d = QuantizedAvgPool1d
IQuantizedAvgPool2d = QuantizedAvgPool2d
IQuantizedAvgPool3d = QuantizedAvgPool3d
IQuantizedAdaptiveAvgPool1d = QuantizedAdaptiveAvgPool2d
IQuantizedAdaptiveAvgPool2d = QuantizedAdaptiveAvgPool2d
IQuantizedAdaptiveAvgPool3d = QuantizedAdaptiveAvgPool2d
IQuantizedBatchNorm1d = QuantizedBatchNorm1d
IQuantizedBatchNorm2d = QuantizedBatchNorm2d
IQuantizedBatchNorm3d = QuantizedBatchNorm3d
IQuantizedLayerNorm = QuantizedLayerNorm
IQuantizedInstanceNorm1d = QuantizedInstanceNorm1d
IQuantizedInstanceNorm2d = QuantizedInstanceNorm2d
IQuantizedInstanceNorm3d = QuantizedInstanceNorm3d
IQuantizedGroupNorm = QuantizedGroupNorm
IQuantizedMultiheadAttention = QuantizedMultiheadAttention
IQuantizedDropout = QuantizedDropout
IQuantizedReLU = QuantizedReLU
IQuantizedSigmoid = QuantizedSigmoid
IQuantizedTanh = QuantizedTanh
IQuantizedLeakyReLU = QuantizedLeakyReLU
IQuantizedSoftmax = QuantizedSoftmax
IQuantizedGELU = QuantizedGELU
IQuantizedELU = QuantizedELU
IQuantizedPReLU = QuantizedPReLU
IQuantizedSiLU = QuantizedSiLU
IQuantizedEmbedding = QuantizedEmbedding


__all__ = [name for name in globals() if name.startswith(('Chop', 'Quantized', 'IQuantized')) or name.endswith('quantization')]
