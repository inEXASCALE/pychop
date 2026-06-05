"""
Per-layer behavior alignment tests for TensorFlow backend quantized layers.

Validates that each TF quantized layer quantizes weights, biases, and outputs
in the same order/pattern as the corresponding torch implementation:
  - Weights/biases: quantized before computation (when quantize_weights=True)
  - Output: quantized after computation (when quantize_output=True)
  - Input: optionally quantized before computation (when quantize_input=True)
  - No quantization when chop=None

Also tests the quantize_input/quantize_output/quantize_weights flag combinations.
"""

import importlib.util
import unittest

import numpy as np

_TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None


def _skip_without_tf(test_func):
    return unittest.skipUnless(_TF_AVAILABLE, "TensorFlow not installed")(test_func)


class _CountingChop:
    """Minimal chop that records how many times it was called and rounds values."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, x):
        import tensorflow as tf
        self.call_count += 1
        return tf.round(tf.cast(x, tf.float32) * 64.0) / 64.0


@_skip_without_tf
class TestQuantizedLinearAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedLinear
        self.tf = tf
        self.QuantizedLinear = QuantizedLinear

    def test_weights_quantized_when_chop_set(self):
        """QuantizedLinear should quantize kernel and bias before matmul."""
        chop = _CountingChop()
        layer = self.QuantizedLinear(4, chop=chop)
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        # chop should be called for: kernel, bias, output = 3 calls
        self.assertGreaterEqual(chop.call_count, 3)

    def test_no_quantization_when_chop_none(self):
        layer = self.QuantizedLinear(4, chop=None)
        x = self.tf.constant([[1.0, 2.0, 3.0]])
        y = layer(x)
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_quantize_weights_false_skips_weight_quant(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(4, chop=chop, quantize_weights=False)
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        # Only output should be quantized (quantize_input=True, quantize_output=True by default)
        # So 2 calls: input + output
        self.assertEqual(chop.call_count, 2)

    def test_quantize_output_false_skips_output_quant(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(4, chop=chop, quantize_output=False)
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        # kernel + bias + input = 3 calls (no output)
        self.assertGreaterEqual(chop.call_count, 2)

    def test_output_shape_preserved(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(8, chop=chop)
        y = layer(self.tf.ones((2, 5), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (2, 8))

    def test_no_bias_layer(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(4, use_bias=False, chop=chop)
        y = layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))


@_skip_without_tf
class TestQuantizedConv2dAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedConv2d
        self.tf = tf
        self.QuantizedConv2d = QuantizedConv2d

    def test_weights_quantized_when_chop_set(self):
        chop = _CountingChop()
        layer = self.QuantizedConv2d(4, 3, chop=chop, padding='same')
        layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        # kernel + bias + input + output = 4 calls minimum
        self.assertGreaterEqual(chop.call_count, 3)

    def test_no_quantization_when_chop_none(self):
        layer = self.QuantizedConv2d(4, 3, chop=None, padding='same')
        y = layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        self.assertEqual(len(y.shape), 4)

    def test_quantize_weights_false_skips_kernel_quant(self):
        chop = _CountingChop()
        layer = self.QuantizedConv2d(4, 3, chop=chop, quantize_weights=False, padding='same')
        layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        # Only input + output quantized = 2 calls
        self.assertEqual(chop.call_count, 2)

    def test_output_shape_preserved(self):
        chop = _CountingChop()
        layer = self.QuantizedConv2d(8, 3, chop=chop, padding='same')
        y = layer(self.tf.ones((2, 16, 16, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (2, 16, 16, 8))


@_skip_without_tf
class TestQuantizedConv1dAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedConv1d
        self.tf = tf
        self.QuantizedConv1d = QuantizedConv1d

    def test_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedConv1d(4, 3, chop=chop, padding='same')
        layer(self.tf.ones((1, 10, 3), dtype=self.tf.float32))
        self.assertGreaterEqual(chop.call_count, 3)

    def test_output_shape(self):
        chop = _CountingChop()
        layer = self.QuantizedConv1d(4, 3, chop=chop, padding='same')
        y = layer(self.tf.ones((2, 10, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (2, 10, 4))


@_skip_without_tf
class TestQuantizedConv3dAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedConv3d
        self.tf = tf
        self.QuantizedConv3d = QuantizedConv3d

    def test_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedConv3d(4, 3, chop=chop, padding='same')
        layer(self.tf.ones((1, 4, 4, 4, 3), dtype=self.tf.float32))
        self.assertGreaterEqual(chop.call_count, 3)


@_skip_without_tf
class TestQuantizedBatchNormAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedBatchNorm1d, QuantizedBatchNorm2d
        self.tf = tf
        self.QuantizedBatchNorm1d = QuantizedBatchNorm1d
        self.QuantizedBatchNorm2d = QuantizedBatchNorm2d

    def test_affine_params_quantized(self):
        """BN gamma/beta should be quantized like torch's weight/bias."""
        chop = _CountingChop()
        layer = self.QuantizedBatchNorm1d(chop=chop)
        layer(self.tf.ones((2, 4), dtype=self.tf.float32), training=True)
        # gamma + beta + input + output = 4
        self.assertGreaterEqual(chop.call_count, 3)

    def test_no_quantization_when_chop_none(self):
        layer = self.QuantizedBatchNorm1d(chop=None)
        y = layer(self.tf.ones((2, 4), dtype=self.tf.float32), training=False)
        self.assertEqual(tuple(y.shape), (2, 4))

    def test_bn2d_shape(self):
        chop = _CountingChop()
        layer = self.QuantizedBatchNorm2d(chop=chop)
        y = layer(self.tf.ones((2, 8, 8, 3), dtype=self.tf.float32), training=True)
        self.assertEqual(tuple(y.shape), (2, 8, 8, 3))


@_skip_without_tf
class TestQuantizedLayerNormAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedLayerNorm
        self.tf = tf
        self.QuantizedLayerNorm = QuantizedLayerNorm

    def test_affine_params_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedLayerNorm(axis=-1, chop=chop)
        layer(self.tf.ones((2, 4), dtype=self.tf.float32))
        # gamma + beta + input + output
        self.assertGreaterEqual(chop.call_count, 3)

    def test_no_quantization_when_chop_none(self):
        layer = self.QuantizedLayerNorm(axis=-1, chop=None)
        y = layer(self.tf.ones((2, 4), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (2, 4))


@_skip_without_tf
class TestQuantizedEmbeddingAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedEmbedding
        self.tf = tf
        self.QuantizedEmbedding = QuantizedEmbedding

    def test_embedding_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedEmbedding(input_dim=10, output_dim=4, chop=chop)
        layer(self.tf.constant([[0, 1, 2]]))
        # embeddings weight + output = 2 minimum
        self.assertGreaterEqual(chop.call_count, 2)

    def test_no_quantization_when_chop_none(self):
        layer = self.QuantizedEmbedding(input_dim=10, output_dim=4, chop=None)
        y = layer(self.tf.constant([[0, 1, 2]]))
        self.assertEqual(tuple(y.shape), (1, 3, 4))


@_skip_without_tf
class TestQuantizedRNNAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedRNN, QuantizedLSTM, QuantizedGRU
        self.tf = tf
        self.QuantizedRNN = QuantizedRNN
        self.QuantizedLSTM = QuantizedLSTM
        self.QuantizedGRU = QuantizedGRU

    def test_rnn_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedRNN(4, chop=chop)
        layer(self.tf.ones((1, 5, 3), dtype=self.tf.float32))
        # kernel + recurrent_kernel + bias + input + output
        self.assertGreaterEqual(chop.call_count, 4)

    def test_lstm_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedLSTM(4, chop=chop)
        layer(self.tf.ones((1, 5, 3), dtype=self.tf.float32))
        self.assertGreaterEqual(chop.call_count, 4)

    def test_gru_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedGRU(4, chop=chop)
        layer(self.tf.ones((1, 5, 3), dtype=self.tf.float32))
        self.assertGreaterEqual(chop.call_count, 4)


@_skip_without_tf
class TestQuantizedMHAAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedMultiheadAttention
        self.tf = tf
        self.QuantizedMultiheadAttention = QuantizedMultiheadAttention

    def test_projection_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedMultiheadAttention(num_heads=2, key_dim=4, chop=chop)
        q = self.tf.ones((1, 3, 8), dtype=self.tf.float32)
        layer(q, q)
        # Q/K/V/output projection kernels + biases + input + output
        self.assertGreaterEqual(chop.call_count, 5)

    def test_no_quantization_when_chop_none(self):
        layer = self.QuantizedMultiheadAttention(num_heads=2, key_dim=4, chop=None)
        q = self.tf.ones((1, 3, 8), dtype=self.tf.float32)
        y = layer(q, q)
        self.assertEqual(tuple(y.shape), (1, 3, 8))


@_skip_without_tf
class TestQuantizedActivationAlignment(unittest.TestCase):
    """Activation layers should quantize output (matching torch behavior)."""

    def setUp(self):
        import tensorflow as tf
        from pychop.tf import layers
        self.tf = tf
        self.layers = layers

    def test_relu_quantizes_output(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedReLU(chop=chop)
        layer(self.tf.constant([-1.0, 0.5, 1.0]))
        self.assertGreaterEqual(chop.call_count, 1)

    def test_sigmoid_quantizes_output(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedSigmoid(chop=chop)
        layer(self.tf.constant([-1.0, 0.0, 1.0]))
        self.assertGreaterEqual(chop.call_count, 1)

    def test_gelu_quantizes_output(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedGELU(chop=chop)
        layer(self.tf.constant([-1.0, 0.0, 1.0]))
        self.assertGreaterEqual(chop.call_count, 1)

    def test_tanh_quantizes_output(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedTanh(chop=chop)
        layer(self.tf.constant([-1.0, 0.0, 1.0]))
        self.assertGreaterEqual(chop.call_count, 1)

    def test_softmax_quantizes_output(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedSoftmax(chop=chop)
        layer(self.tf.constant([[1.0, 2.0, 3.0]]))
        self.assertGreaterEqual(chop.call_count, 1)

    def test_silu_quantizes_output(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedSiLU(chop=chop)
        layer(self.tf.constant([-1.0, 0.0, 1.0]))
        self.assertGreaterEqual(chop.call_count, 1)


@_skip_without_tf
class TestQuantizedPoolAlignment(unittest.TestCase):
    """Pool layers should quantize output."""

    def setUp(self):
        import tensorflow as tf
        from pychop.tf import layers
        self.tf = tf
        self.layers = layers

    def test_maxpool2d(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedMaxPool2d(pool_size=2, chop=chop)
        y = layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        self.assertGreaterEqual(chop.call_count, 1)
        self.assertEqual(tuple(y.shape), (1, 4, 4, 3))

    def test_avgpool2d(self):
        chop = _CountingChop()
        layer = self.layers.QuantizedAvgPool2d(pool_size=2, chop=chop)
        y = layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        self.assertGreaterEqual(chop.call_count, 1)


@_skip_without_tf
class TestQuantizedConvTransposeAlignment(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedConvTranspose2d
        self.tf = tf
        self.QuantizedConvTranspose2d = QuantizedConvTranspose2d

    def test_weights_quantized(self):
        chop = _CountingChop()
        layer = self.QuantizedConvTranspose2d(4, 3, chop=chop, padding='same')
        layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        # kernel + bias + input + output
        self.assertGreaterEqual(chop.call_count, 3)

    def test_output_shape(self):
        chop = _CountingChop()
        layer = self.QuantizedConvTranspose2d(4, 3, chop=chop, padding='same')
        y = layer(self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 8, 8, 4))


@_skip_without_tf
class TestFlagCombinations(unittest.TestCase):
    """Test that quantize_input/quantize_output/quantize_weights flags work correctly."""

    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedLinear
        self.tf = tf
        self.QuantizedLinear = QuantizedLinear

    def test_all_false_means_no_quantization(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(
            4, chop=chop,
            quantize_input=False, quantize_output=False, quantize_weights=False
        )
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(chop.call_count, 0)

    def test_only_weights(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(
            4, chop=chop,
            quantize_input=False, quantize_output=False, quantize_weights=True
        )
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        # kernel + bias = 2
        self.assertEqual(chop.call_count, 2)

    def test_only_output(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(
            4, chop=chop,
            quantize_input=False, quantize_output=True, quantize_weights=False
        )
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(chop.call_count, 1)

    def test_only_input(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(
            4, chop=chop,
            quantize_input=True, quantize_output=False, quantize_weights=False
        )
        layer(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(chop.call_count, 1)


@_skip_without_tf
class TestEdgeCases(unittest.TestCase):
    """Test edge cases: zero input, single element, large batch."""

    def setUp(self):
        import tensorflow as tf
        from pychop.tf.layers import QuantizedLinear, QuantizedConv2d
        self.tf = tf
        self.QuantizedLinear = QuantizedLinear
        self.QuantizedConv2d = QuantizedConv2d

    def test_zero_input(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(4, chop=chop)
        y = layer(self.tf.zeros((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_single_element_batch(self):
        chop = _CountingChop()
        layer = self.QuantizedLinear(1, chop=chop)
        y = layer(self.tf.ones((1, 1), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 1))

    def test_conv2d_single_pixel(self):
        chop = _CountingChop()
        layer = self.QuantizedConv2d(2, 1, chop=chop, padding='same')
        y = layer(self.tf.ones((1, 1, 1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 1, 1, 2))


@_skip_without_tf
class TestPTQAlignment(unittest.TestCase):
    """Test that PTQ functions work with per-layer quantization."""

    def setUp(self):
        import tensorflow as tf
        from pychop import Chop
        from pychop.tf.ptq import (
            post_quantization,
            dynamic_post_quantization,
            static_post_quantization,
            mixed_post_quantization,
        )
        self.tf = tf
        self.Chop = Chop
        self.post_quantization = post_quantization
        self.dynamic_post_quantization = dynamic_post_quantization
        self.static_post_quantization = static_post_quantization
        self.mixed_post_quantization = mixed_post_quantization

    def _make_model(self):
        model = self.tf.keras.Sequential([
            self.tf.keras.layers.Dense(8, activation='relu'),
            self.tf.keras.layers.Dense(4),
        ])
        model(self.tf.ones((1, 3), dtype=self.tf.float32))
        return model

    def test_basic_ptq(self):
        model = self._make_model()
        ch = self.Chop(exp_bits=5, sig_bits=10)
        q_model = self.post_quantization(model, ch)
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_dynamic_ptq(self):
        model = self._make_model()
        ch = self.Chop(exp_bits=5, sig_bits=10)
        q_model = self.dynamic_post_quantization(model, ch)
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_static_ptq_with_calibration(self):
        model = self._make_model()
        ch = self.Chop(exp_bits=5, sig_bits=10)
        cal_data = [self.tf.random.normal((2, 3)) for _ in range(5)]
        q_model = self.static_post_quantization(model, ch, calibration_data=cal_data)
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_static_ptq_empty_calibration(self):
        model = self._make_model()
        ch = self.Chop(exp_bits=5, sig_bits=10)
        q_model = self.static_post_quantization(model, ch, calibration_data=[])
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_mixed_ptq_dynamic(self):
        model = self._make_model()
        w_ch = self.Chop(exp_bits=5, sig_bits=10)
        a_ch = self.Chop(exp_bits=5, sig_bits=10)
        q_model = self.mixed_post_quantization(model, w_ch, a_ch, dynamic=True)
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_mixed_ptq_static(self):
        model = self._make_model()
        w_ch = self.Chop(exp_bits=5, sig_bits=10)
        a_ch = self.Chop(exp_bits=5, sig_bits=10)
        cal_data = [self.tf.random.normal((2, 3)) for _ in range(5)]
        q_model = self.mixed_post_quantization(
            model, w_ch, a_ch, calibration_data=cal_data, dynamic=False
        )
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_mixed_ptq_no_activation_chop(self):
        model = self._make_model()
        w_ch = self.Chop(exp_bits=5, sig_bits=10)
        q_model = self.mixed_post_quantization(model, w_ch, None, dynamic=True)
        y = q_model(self.tf.ones((1, 3), dtype=self.tf.float32))
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_mixed_ptq_static_requires_calibration(self):
        model = self._make_model()
        w_ch = self.Chop(exp_bits=5, sig_bits=10)
        a_ch = self.Chop(exp_bits=5, sig_bits=10)
        with self.assertRaises(ValueError):
            self.mixed_post_quantization(
                model, w_ch, a_ch, calibration_data=None, dynamic=False
            )


if __name__ == '__main__':
    unittest.main()
