import importlib.util
import unittest

import numpy as np

import pychop
from pychop import layers as frontend_layers
from pychop import ptq as frontend_ptq
from pychop.set_backend import _VALID_BACKENDS
from pychop.utils import detect_array_type, to_numpy_array


class TensorFlowBackendSmokeTests(unittest.TestCase):
    def tearDown(self):
        pychop.backend('auto')

    def test_backend_registry_includes_tensorflow(self):
        self.assertIn('tensorflow', _VALID_BACKENDS)
        self.assertIn('tf', _VALID_BACKENDS)

    def test_tf_alias_falls_back_without_tensorflow(self):
        pychop.backend('tf')
        if importlib.util.find_spec('tensorflow') is None:
            self.assertEqual(pychop.get_backend(), 'numpy')
        else:
            self.assertEqual(pychop.get_backend(), 'tensorflow')

    def test_numpy_utility_still_works(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqual(detect_array_type(arr), 'numpy')
        np.testing.assert_array_equal(to_numpy_array(arr), arr)

    def test_auto_backend_model_resolution_prefers_tensorflow(self):
        class DummyTFModel:
            pass

        DummyTFModel.__module__ = 'tensorflow.keras.engine.training'
        model = DummyTFModel()
        self.assertEqual(frontend_layers._resolve_backend_for_model(model), 'tensorflow')
        self.assertEqual(frontend_ptq._resolve_backend_for_model(model), 'tensorflow')

    def test_tensorflow_lightchop_and_fixed_point_surface_methods(self):
        if importlib.util.find_spec('tensorflow') is None:
            self.skipTest("TensorFlow not installed")

        from pychop.tf.lightchop import LightChop_
        from pychop.tf.fixed_point import FPRound_

        ch = LightChop_(exp_bits=5, sig_bits=10)
        fp = FPRound_(ibits=4, fbits=4)

        for method in ('sin', 'add', 'sum', 'round', 'clip', 'ldexp', 'frexp', 'modf'):
            self.assertTrue(hasattr(ch, method), f"LightChop_ missing {method}")
            self.assertTrue(hasattr(fp, method), f"FPRound_ missing {method}")

    def test_tensorflow_integer_calibration_state(self):
        if importlib.util.find_spec('tensorflow') is None:
            self.skipTest("TensorFlow not installed")

        import tensorflow as tf
        from pychop.tf.integer import Chopi_

        ch = Chopi_(bits=8)
        self.assertFalse(ch.is_calibrated)
        ch.calibrate(tf.constant([0.1, -0.2, 0.3], dtype=tf.float32))
        self.assertTrue(ch.is_calibrated)

    def test_tensorflow_static_ptq_accepts_empty_calibration(self):
        if importlib.util.find_spec('tensorflow') is None:
            self.skipTest("TensorFlow not installed")

        import tensorflow as tf
        from pychop import Chop
        from pychop.tf.ptq import static_post_quantization

        model = tf.keras.Sequential([tf.keras.layers.Dense(4)])
        model(tf.ones((1, 3), dtype=tf.float32))
        ch = Chop(exp_bits=5, sig_bits=10)
        wrapped = static_post_quantization(model, ch, calibration_data=[])
        y = wrapped(tf.ones((1, 3), dtype=tf.float32), training=False)
        self.assertEqual(tuple(y.shape), (1, 4))


if __name__ == '__main__':
    unittest.main()
