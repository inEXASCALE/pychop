"""
Test script for PyTorch backend PTQ functions:
  - post_quantization       (weight-only, 原有)
  - static_post_quantization  (权重 + 静态激活量化)
  - dynamic_post_quantization (权重 + 动态激活量化)
  - mixed_post_quantization   (W/A 独立量化, e.g. W8A8)

Run:
    python -m pytest tests/test_ptq_torch.py -v
    or
    python tests/test_ptq_torch.py
"""

import os
os.environ["chop_backend"] = "torch"

import copy
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pychop
pychop.backend("torch")

from pychop import Chop, Chopi, Chopf
from pychop.tch.layers import (
    post_quantization,
    static_post_quantization,
    dynamic_post_quantization,
    mixed_post_quantization,
)


# ===================================================================
# Helper: simple CNN for testing
# ===================================================================

class SimpleCNN(nn.Module):
    """Minimal CNN: Conv2d -> BN -> ReLU -> Linear."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _make_model_and_data(num_samples=64, batch_size=16):
    """Create a trained model and a small calibration DataLoader."""
    torch.manual_seed(42)
    model = SimpleCNN(num_classes=10)
    model.eval()

    # Synthetic data
    images = torch.randn(num_samples, 1, 8, 8)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return model, loader, images


# ===================================================================
# Tests
# ===================================================================

class TestPostQuantization(unittest.TestCase):
    """Test the original weight-only post_quantization."""

    def setUp(self):
        self.model, self.loader, self.images = _make_model_and_data()

    def test_basic_chop(self):
        """Weight-only PTQ with Chop (floating-point quantizer)."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = post_quantization(self.model, chop, eval_mode=True, verbose=False)

        # Must return nn.Module
        self.assertIsInstance(q_model, nn.Module)

        # Weights should be modified
        orig_w = self.model.conv1.weight.data.clone()
        q_w = q_model.conv1.weight.data
        # At least some values should differ (quantization changes values)
        self.assertFalse(torch.equal(orig_w, q_w),
                         "Weights should be different after quantization")

        # Shape preserved
        self.assertEqual(orig_w.shape, q_w.shape)

        # Output shape should be correct
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_original_unchanged(self):
        """Original model should NOT be modified."""
        orig_w = self.model.conv1.weight.data.clone()
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        _ = post_quantization(self.model, chop)
        self.assertTrue(torch.equal(self.model.conv1.weight.data, orig_w))

    def test_chopf(self):
        """Weight-only PTQ with Chopf (fixed-point quantizer)."""
        chop = Chopf(ibits=4, fbits=4, rmode=1)
        q_model = post_quantization(self.model, chop, verbose=False)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_chopi(self):
        """Weight-only PTQ with Chopi (integer quantizer)."""
        chop = Chopi(bits=8, symmetric=True)
        q_model = post_quantization(self.model, chop, verbose=False)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_buffers_preserved(self):
        """BN running_mean / running_var should NOT be quantized."""
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        orig_running_mean = self.model.bn1.running_mean.clone()
        q_model = post_quantization(self.model, chop, verbose=False)
        self.assertTrue(
            torch.equal(q_model.bn1.running_mean, orig_running_mean),
            "BN running_mean should be preserved (not quantized)"
        )


class TestStaticPostQuantization(unittest.TestCase):
    """Test static_post_quantization (weight + calibrated activation)."""

    def setUp(self):
        self.model, self.loader, self.images = _make_model_and_data()

    def test_basic(self):
        """Static PTQ should run and produce correct output shape."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = static_post_quantization(
            self.model, chop,
            calibration_data=self.loader,
            eval_mode=True, verbose=False,
        )
        self.assertIsInstance(q_model, nn.Module)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_weights_quantized(self):
        """Weights should differ from original after static PTQ."""
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        q_model = static_post_quantization(
            self.model, chop,
            calibration_data=self.loader, verbose=False,
        )
        self.assertFalse(
            torch.equal(self.model.conv1.weight.data, q_model.conv1.weight.data)
        )

    def test_activation_hooks_registered(self):
        """Hooks should be registered on target layers."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = static_post_quantization(
            self.model, chop,
            calibration_data=self.loader, verbose=False,
        )
        # After static PTQ, at least some modules should have forward hooks
        has_hooks = any(
            len(m._forward_hooks) > 0
            for m in q_model.modules()
        )
        self.assertTrue(has_hooks, "Static PTQ should register activation hooks")

    def test_output_differs_from_fp(self):
        """Output of quantized model should differ from FP model."""
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        q_model = static_post_quantization(
            self.model, chop,
            calibration_data=self.loader, verbose=False,
        )
        with torch.no_grad():
            fp_out = self.model(self.images[:4])
            q_out = q_model(self.images[:4])
        self.assertFalse(torch.allclose(fp_out, q_out, atol=1e-7),
                         "Quantized output should differ from FP output")

    def test_verbose(self):
        """Verbose mode should not crash."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = static_post_quantization(
            self.model, chop,
            calibration_data=self.loader,
            verbose=True,
        )
        self.assertIsInstance(q_model, nn.Module)

    def test_with_chopi(self):
        """Static PTQ with integer quantizer."""
        chop = Chopi(bits=8, symmetric=True)
        q_model = static_post_quantization(
            self.model, chop,
            calibration_data=self.loader, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))


class TestDynamicPostQuantization(unittest.TestCase):
    """Test dynamic_post_quantization (weight + per-inference activation)."""

    def setUp(self):
        self.model, self.loader, self.images = _make_model_and_data()

    def test_basic(self):
        """Dynamic PTQ should run and produce correct output shape."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = dynamic_post_quantization(
            self.model, chop, eval_mode=True, verbose=False,
        )
        self.assertIsInstance(q_model, nn.Module)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_no_calibration_needed(self):
        """Dynamic PTQ should NOT require calibration data."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        # No calibration_data argument at all
        q_model = dynamic_post_quantization(self.model, chop, verbose=False)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_weights_quantized(self):
        """Weights should differ from original after dynamic PTQ."""
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        q_model = dynamic_post_quantization(self.model, chop, verbose=False)
        self.assertFalse(
            torch.equal(self.model.conv1.weight.data, q_model.conv1.weight.data)
        )

    def test_hooks_registered(self):
        """Forward hooks should be registered for dynamic activation quantization."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = dynamic_post_quantization(self.model, chop, verbose=False)
        has_hooks = any(
            len(m._forward_hooks) > 0
            for m in q_model.modules()
        )
        self.assertTrue(has_hooks)

    def test_output_differs_from_fp(self):
        """Output should differ from FP model."""
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        q_model = dynamic_post_quantization(self.model, chop, verbose=False)
        with torch.no_grad():
            fp_out = self.model(self.images[:4])
            q_out = q_model(self.images[:4])
        self.assertFalse(torch.allclose(fp_out, q_out, atol=1e-7))

    def test_deterministic(self):
        """Two calls with same input should give same output (rmode != stochastic)."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = dynamic_post_quantization(self.model, chop, verbose=False)
        with torch.no_grad():
            out1 = q_model(self.images[:4])
            out2 = q_model(self.images[:4])
        self.assertTrue(torch.equal(out1, out2))

    def test_with_chopf(self):
        """Dynamic PTQ with fixed-point quantizer."""
        chop = Chopf(ibits=4, fbits=4, rmode=1)
        q_model = dynamic_post_quantization(self.model, chop, verbose=False)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))


class TestMixedPostQuantization(unittest.TestCase):
    """Test mixed_post_quantization (separate W/A quantizers)."""

    def setUp(self):
        self.model, self.loader, self.images = _make_model_and_data()

    # ---------- Dynamic mode ----------

    def test_w8a8_dynamic(self):
        """W8A8 dynamic: both weight and activation quantized."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop, dynamic=True, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_w4a8_dynamic(self):
        """W4A8 dynamic: aggressive weight, moderate activation."""
        w_chop = Chop(exp_bits=2, sig_bits=1, rmode=1)  # ~4-bit
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)  # ~8-bit
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop, dynamic=True, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_weight_only(self):
        """activation_chop=None → weight-only (like post_quantization)."""
        w_chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, None, verbose=False,
        )
        # No activation hooks should be registered
        has_hooks = any(
            len(m._forward_hooks) > 0
            for m in q_model.modules()
        )
        self.assertFalse(has_hooks, "No activation hooks when activation_chop=None")
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_activation_only(self):
        """weight_chop=None → FP weights, quantized activations."""
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mixed_post_quantization(
            self.model, None, a_chop, dynamic=True, verbose=False,
        )
        # Weights should be unchanged
        self.assertTrue(
            torch.equal(
                self.model.conv1.weight.data,
                q_model.conv1.weight.data,
            )
        )
        # But activation hooks should exist
        has_hooks = any(
            len(m._forward_hooks) > 0
            for m in q_model.modules()
        )
        self.assertTrue(has_hooks)
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_none_none(self):
        """Both None → equivalent to deepcopy, no quantization."""
        q_model = mixed_post_quantization(
            self.model, None, None, verbose=False,
        )
        with torch.no_grad():
            orig_out = self.model(self.images[:4])
            q_out = q_model(self.images[:4])
        self.assertTrue(torch.allclose(orig_out, q_out, atol=1e-7))

    # ---------- Static mode ----------

    def test_w8a8_static(self):
        """W8A8 static: with calibration data."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop,
            calibration_data=self.loader,
            dynamic=False, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_static_requires_calibration(self):
        """static mode + activation_chop without calibration_data → ValueError."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        with self.assertRaises(ValueError):
            mixed_post_quantization(
                self.model, w_chop, a_chop,
                calibration_data=None, dynamic=False,
            )

    def test_static_activation_none_no_error(self):
        """static mode + activation_chop=None → no calibration needed."""
        w_chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, None,
            calibration_data=None, dynamic=False, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    # ---------- Integer quantizers ----------

    def test_mixed_integer(self):
        """W8A4 with Chopi (integer quantizers)."""
        w_chop = Chopi(bits=8, symmetric=True)
        a_chop = Chopi(bits=4, symmetric=False)
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop, dynamic=True, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    def test_mixed_w_int_a_float(self):
        """Wint8 + A-fp8: cross-type quantizers."""
        w_chop = Chopi(bits=8, symmetric=True)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop, dynamic=True, verbose=False,
        )
        with torch.no_grad():
            out = q_model(self.images[:4])
        self.assertEqual(out.shape, (4, 10))

    # ---------- Verbose ----------

    def test_verbose_dynamic(self):
        """Verbose mode (dynamic) should not crash."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop, dynamic=True, verbose=True,
        )
        self.assertIsInstance(q_model, nn.Module)

    def test_verbose_static(self):
        """Verbose mode (static) should not crash."""
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mixed_post_quantization(
            self.model, w_chop, a_chop,
            calibration_data=self.loader,
            dynamic=False, verbose=True,
        )
        self.assertIsInstance(q_model, nn.Module)


class TestOriginalModelUnchanged(unittest.TestCase):
    """All PTQ functions must NOT modify the original model."""

    def setUp(self):
        self.model, self.loader, self.images = _make_model_and_data()
        self.orig_state = copy.deepcopy(self.model.state_dict())

    def _assert_unchanged(self):
        for key, orig_val in self.orig_state.items():
            new_val = self.model.state_dict()[key]
            self.assertTrue(torch.equal(orig_val, new_val),
                            f"Original model's {key} was modified!")

    def test_post_quantization(self):
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        _ = post_quantization(self.model, chop)
        self._assert_unchanged()

    def test_static(self):
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        _ = static_post_quantization(self.model, chop, self.loader)
        self._assert_unchanged()

    def test_dynamic(self):
        chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        _ = dynamic_post_quantization(self.model, chop)
        self._assert_unchanged()

    def test_mixed(self):
        w_chop = Chop(exp_bits=3, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        _ = mixed_post_quantization(self.model, w_chop, a_chop)
        self._assert_unchanged()


class TestFrontendDispatch(unittest.TestCase):
    """Test that pychop/layers.py frontend dispatches work for torch backend."""

    def setUp(self):
        os.environ["chop_backend"] = "torch"
        self.model, self.loader, self.images = _make_model_and_data()

    def test_dispatch_post_quantization(self):
        from pychop.layers import post_quantization as ptq_dispatch
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = ptq_dispatch(self.model, chop, verbose=False)
        self.assertIsInstance(q_model, nn.Module)

    def test_dispatch_static(self):
        from pychop.layers import static_post_quantization as sptq_dispatch
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = sptq_dispatch(self.model, chop, self.loader, verbose=False)
        self.assertIsInstance(q_model, nn.Module)

    def test_dispatch_dynamic(self):
        from pychop.layers import dynamic_post_quantization as dptq_dispatch
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        q_model = dptq_dispatch(self.model, chop, verbose=False)
        self.assertIsInstance(q_model, nn.Module)

    def test_dispatch_mixed(self):
        from pychop.layers import mixed_post_quantization as mptq_dispatch
        w_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        a_chop = Chop(exp_bits=4, sig_bits=3, rmode=1)
        q_model = mptq_dispatch(self.model, w_chop, a_chop, verbose=False)
        self.assertIsInstance(q_model, nn.Module)


class TestCompareQuantizationMethods(unittest.TestCase):
    """Sanity check: compare outputs of different PTQ methods."""

    def setUp(self):
        self.model, self.loader, self.images = _make_model_and_data()
        self.chop = Chop(exp_bits=3, sig_bits=3, rmode=1)

    def test_static_vs_dynamic_differ(self):
        """Static and dynamic PTQ should generally produce different outputs."""
        q_static = static_post_quantization(
            self.model, self.chop, self.loader, verbose=False,
        )
        q_dynamic = dynamic_post_quantization(
            self.model, self.chop, verbose=False,
        )
        with torch.no_grad():
            out_s = q_static(self.images[:4])
            out_d = q_dynamic(self.images[:4])

        # They CAN be equal in degenerate cases, but usually differ
        # We only check they both produce valid output
        self.assertEqual(out_s.shape, (4, 10))
        self.assertEqual(out_d.shape, (4, 10))

    def test_weight_only_vs_mixed_none(self):
        """post_quantization should match mixed(w, None)."""
        q_orig = post_quantization(self.model, self.chop, verbose=False)
        q_mixed = mixed_post_quantization(
            self.model, self.chop, None, verbose=False,
        )
        with torch.no_grad():
            out_orig = q_orig(self.images[:4])
            out_mixed = q_mixed(self.images[:4])
        # Should produce same output (both weight-only, no activation hooks)
        self.assertTrue(torch.allclose(out_orig, out_mixed, atol=1e-6))

    def test_all_methods_finite(self):
        """All PTQ methods should produce finite (non-NaN, non-Inf) outputs."""
        chop = Chop(exp_bits=5, sig_bits=10, rmode=1)
        methods = [
            ("post_quantization",
             lambda: post_quantization(self.model, chop)),
            ("static_post_quantization",
             lambda: static_post_quantization(self.model, chop, self.loader)),
            ("dynamic_post_quantization",
             lambda: dynamic_post_quantization(self.model, chop)),
            ("mixed_post_quantization (W8A8)",
             lambda: mixed_post_quantization(self.model, chop, chop)),
            ("mixed_post_quantization (W-only)",
             lambda: mixed_post_quantization(self.model, chop, None)),
            ("mixed_post_quantization (A-only)",
             lambda: mixed_post_quantization(self.model, None, chop)),
        ]
        for name, factory in methods:
            with self.subTest(method=name):
                q_model = factory()
                with torch.no_grad():
                    out = q_model(self.images[:4])
                self.assertTrue(torch.isfinite(out).all(),
                                f"{name} produced non-finite output")


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"pychop backend: {pychop.get_backend()}")
    print()
    unittest.main(verbosity=2)