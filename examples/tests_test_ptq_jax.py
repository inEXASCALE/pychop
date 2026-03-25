"""
Test script for JAX backend PTQ functions:
  - post_quantization         (weight-only, 原有)
  - static_post_quantization  (权重 + 静态激活量化)
  - dynamic_post_quantization (权重 + 动态激活量化)
  - mixed_post_quantization   (W/A 独立量化)

"""

import os
os.environ["chop_backend"] = "jax"

import unittest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from flax import linen as nn
    from flax.training import train_state
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import pychop
pychop.backend("jax")

from pychop import Chop, Chopi


# ===================================================================
# Helper: simple Flax CNN
# ===================================================================

if HAS_JAX:

    class SimpleFlaxCNN(nn.Module):
        """Minimal Flax CNN: Conv -> BN -> ReLU -> Dense."""
        num_classes: int = 10

        @nn.compact
        def __call__(self, x, training: bool = False):
            x = nn.Conv(features=8, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            # Global average pool
            x = jnp.mean(x, axis=(1, 2))
            x = nn.Dense(features=self.num_classes)(x)
            return x

    def _make_jax_model_and_data(num_samples=64):
        """Create a Flax model, initialized params, and synthetic data."""
        rng = random.PRNGKey(0)
        model = SimpleFlaxCNN(num_classes=10)

        # Init
        dummy = jnp.ones((1, 8, 8, 1))
        variables = model.init(rng, dummy, training=False)
        params = variables['params']
        batch_stats = variables.get('batch_stats', None)

        # Synthetic images
        images = random.normal(rng, (num_samples, 8, 8, 1))
        labels = random.randint(rng, (num_samples,), 0, 10)

        return model, params, batch_stats, images, labels


# ===================================================================
# Tests
# ===================================================================

@unittest.skipUnless(HAS_JAX, "JAX/Flax not installed")
class TestJaxPostQuantization(unittest.TestCase):
    """Test the original weight-only post_quantization for JAX."""

    def setUp(self):
        from pychop.jx.layers import post_quantization
        self.post_quantization = post_quantization
        self.model, self.params, self.batch_stats, self.images, self.labels = (
            _make_jax_model_and_data()
        )

    def test_basic(self):
        """Weight-only PTQ should return quantized params."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        variables = {'params': self.params}
        if self.batch_stats is not None:
            variables['batch_stats'] = self.batch_stats

        result = self.post_quantization(variables, chop, verbose=False)
        self.assertIsInstance(result, dict)
        self.assertIn('params', result)

    def test_params_changed(self):
        """At least some params should be different after quantization."""
        chop = Chop(exp_bits=2, sig_bits=2, rmode=1)
        variables = {'params': self.params}
        if self.batch_stats is not None:
            variables['batch_stats'] = self.batch_stats

        result = self.post_quantization(variables, chop, verbose=False)

        # Compare a specific leaf
        orig_kernel = self.params['Conv_0']['kernel']
        q_kernel = result['params']['Conv_0']['kernel']
        self.assertFalse(jnp.array_equal(orig_kernel, q_kernel),
                         "Params should change after quantization")

    def test_model_runs_after_ptq(self):
        """Model should be callable with quantized params."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        variables = {'params': self.params}
        if self.batch_stats is not None:
            variables['batch_stats'] = self.batch_stats

        result = self.post_quantization(variables, chop, verbose=False)
        q_vars = {'params': result['params']}
        if self.batch_stats is not None:
            q_vars['batch_stats'] = result.get('batch_stats', self.batch_stats)

        out = self.model.apply(q_vars, self.images[:4], training=False)
        self.assertEqual(out.shape, (4, 10))


@unittest.skipUnless(HAS_JAX, "JAX/Flax not installed")
class TestJaxStaticPostQuantization(unittest.TestCase):
    """Test static_post_quantization for JAX."""

    def setUp(self):
        from pychop.jx.layers import static_post_quantization
        self.static_ptq = static_post_quantization
        self.model, self.params, self.batch_stats, self.images, self.labels = (
            _make_jax_model_and_data()
        )
        # Make calibration data as list of batches
        self.cal_data = [self.images[i:i+16] for i in range(0, 64, 16)]

    def test_basic(self):
        """Static PTQ should return dict with params and activation_stats."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.static_ptq(
            self.model, chop,
            calibration_data=self.cal_data,
            verbose=False,
        )
        self.assertIsInstance(result, dict)
        self.assertIn('params', result)

    def test_has_quantized_apply(self):
        """Result should contain a callable quantized_apply."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.static_ptq(
            self.model, chop,
            calibration_data=self.cal_data,
            verbose=False,
        )
        self.assertIn('quantized_apply', result)
        self.assertTrue(callable(result['quantized_apply']))

    def test_verbose(self):
        """Verbose should not crash."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.static_ptq(
            self.model, chop,
            calibration_data=self.cal_data,
            verbose=True,
        )
        self.assertIsInstance(result, dict)


@unittest.skipUnless(HAS_JAX, "JAX/Flax not installed")
class TestJaxDynamicPostQuantization(unittest.TestCase):
    """Test dynamic_post_quantization for JAX."""

    def setUp(self):
        from pychop.jx.layers import dynamic_post_quantization
        self.dynamic_ptq = dynamic_post_quantization
        self.model, self.params, self.batch_stats, self.images, self.labels = (
            _make_jax_model_and_data()
        )

    def test_basic(self):
        """Dynamic PTQ should return dict with params and dynamic_apply."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.dynamic_ptq(self.model, chop, verbose=False)
        self.assertIsInstance(result, dict)
        self.assertIn('params', result)
        self.assertIn('dynamic_apply', result)

    def test_dynamic_apply_callable(self):
        """dynamic_apply should be callable."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.dynamic_ptq(self.model, chop, verbose=False)
        self.assertTrue(callable(result['dynamic_apply']))

    def test_no_calibration_needed(self):
        """Should not require calibration data."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        # This should not raise
        result = self.dynamic_ptq(self.model, chop, verbose=False)
        self.assertIn('params', result)

    def test_verbose(self):
        """Verbose should not crash."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.dynamic_ptq(self.model, chop, verbose=True)
        self.assertIsInstance(result, dict)


@unittest.skipUnless(HAS_JAX, "JAX/Flax not installed")
class TestJaxMixedPostQuantization(unittest.TestCase):
    """Test mixed_post_quantization for JAX."""

    def setUp(self):
        from pychop.jx.layers import mixed_post_quantization
        self.mixed_ptq = mixed_post_quantization
        self.model, self.params, self.batch_stats, self.images, self.labels = (
            _make_jax_model_and_data()
        )
        self.cal_data = [self.images[i:i+16] for i in range(0, 64, 16)]

    def test_w8a8_dynamic(self):
        """W8A8 dynamic should return dict with mixed_apply."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        result = self.mixed_ptq(
            self.model, w_chop, a_chop, dynamic=True, verbose=False,
        )
        self.assertIsInstance(result, dict)
        self.assertIn('params', result)
        self.assertIn('mixed_apply', result)
        self.assertTrue(callable(result['mixed_apply']))

    def test_weight_only(self):
        """activation_chop=None → weight-only."""
        w_chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        result = self.mixed_ptq(
            self.model, w_chop, None, verbose=False,
        )
        self.assertIn('params', result)
        self.assertIn('mixed_apply', result)

    def test_activation_only(self):
        """weight_chop=None → FP weights."""
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        result = self.mixed_ptq(
            self.model, None, a_chop, dynamic=True, verbose=False,
        )
        self.assertIn('params', result)

    def test_static_requires_calibration(self):
        """static mode + activation_chop without calibration → ValueError."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        with self.assertRaises(ValueError):
            self.mixed_ptq(
                self.model, w_chop, a_chop,
                calibration_data=None, dynamic=False,
            )

    def test_w8a8_static(self):
        """W8A8 static with calibration data."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        result = self.mixed_ptq(
            self.model, w_chop, a_chop,
            calibration_data=self.cal_data,
            dynamic=False, verbose=False,
        )
        self.assertIn('params', result)
        self.assertIn('mixed_apply', result)

    def test_none_none(self):
        """Both None → no quantization."""
        result = self.mixed_ptq(
            self.model, None, None, verbose=False,
        )
        self.assertIn('mixed_apply', result)

    def test_verbose_dynamic(self):
        """Verbose dynamic should not crash."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        result = self.mixed_ptq(
            self.model, w_chop, a_chop, dynamic=True, verbose=True,
        )
        self.assertIsInstance(result, dict)

    def test_verbose_static(self):
        """Verbose static should not crash."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        result = self.mixed_ptq(
            self.model, w_chop, a_chop,
            calibration_data=self.cal_data,
            dynamic=False, verbose=True,
        )
        self.assertIsInstance(result, dict)


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    if HAS_JAX:
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    else:
        print("WARNING: JAX/Flax not installed, all tests will be skipped")
    print(f"pychop backend: {pychop.get_backend()}")
    print()
    unittest.main(verbosity=2)