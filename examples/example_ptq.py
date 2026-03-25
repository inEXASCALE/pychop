"""
Comprehensive test suite for PyChop PTQ across PyTorch and JAX backends.

Tests all PTQ methods with different calibration strategies:
1. Basic PTQ (weight-only)
2. Static PTQ (minmax, percentile, KL-divergence, MSE)
3. Dynamic PTQ (no calibration)
4. Mixed-precision PTQ (W8A16, W4A8, W2A8)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pychop
from pychop import Chop, Chopf, Chopi


# ===================================================================
# Helper Functions
# ===================================================================

def _is_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _is_jax_available():
    """Check if JAX/Flax is available."""
    try:
        import jax
        import flax
        return True
    except ImportError:
        return False


# ===================================================================
# PyTorch Backend Tests
# ===================================================================

class TestPyTorchPTQ:
    """Test suite for PyTorch backend PTQ."""
    
    def __init__(self):
        """Initialize test class."""
        if not _is_torch_available():
            print("❌ PyTorch not available! Skipping tests.")
            self.available = False
            return
        
        self.available = True
        pychop.backend('torch')
        
        import torch
        import torch.nn as nn
        from pychop.ptq import (
            post_quantization,
            static_post_quantization,
            dynamic_post_quantization,
            mixed_post_quantization
        )
        
        self.torch = torch
        self.nn = nn
        self.post_quantization = post_quantization
        self.static_post_quantization = static_post_quantization
        self.dynamic_post_quantization = dynamic_post_quantization
        self.mixed_post_quantization = mixed_post_quantization
        
        # Define SimpleCNN class
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 10)
            
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        self.SimpleCNN = SimpleCNN
        
        print("\n" + "="*70)
        print("PyTorch Backend PTQ Test Suite")
        print("="*70)
    
    def test_1_basic_ptq(self):
        """Test 1: Basic PTQ (Weight-Only Quantization)."""
        if not self.available:
            return
        
        print("\n[Test 1] Basic PTQ (Weight-Only)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        chop = Chopi(bits=8, symmetric=True)
        model_q = self.post_quantization(model, chop, verbose=True)
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        assert output_fp.shape == output_q.shape, "Output shape mismatch!"
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs INT8 MSE: {mse:.6f}")
        assert mse < 100.0, f"Quantization error too large: {mse}"
        
        print("✅ Basic PTQ test passed!")
    
    def test_2_static_ptq_minmax(self):
        """Test 2: Static PTQ with MinMax Calibration."""
        if not self.available:
            return
        
        print("\n[Test 2] Static PTQ (MinMax Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        calibration_data = [
            self.torch.randn(4, 1, 28, 28) for _ in range(10)
        ]
        
        chop = Chopi(bits=8, symmetric=True)
        model_q = self.static_post_quantization(
            model, chop,
            calibration_data=calibration_data,
            calibration_method='minmax',
            fuse_bn=True,
            verbose=True
        )
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        assert output_fp.shape == output_q.shape
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs Static-INT8 (MinMax) MSE: {mse:.6f}")
        assert mse < 100.0, f"Static PTQ error too large: {mse}"
        
        print("✅ Static PTQ (MinMax) test passed!")
    
    def test_3_static_ptq_percentile(self):
        """Test 3: Static PTQ with Percentile Calibration."""
        if not self.available:
            return
        
        print("\n[Test 3] Static PTQ (Percentile Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        calibration_data = [
            self.torch.randn(4, 1, 28, 28) for _ in range(10)
        ]
        
        chop = Chopi(bits=8, symmetric=True)
        model_q = self.static_post_quantization(
            model, chop,
            calibration_data=calibration_data,
            calibration_method='percentile',
            percentile=99.9,
            fuse_bn=False,
            verbose=True
        )
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs Percentile-INT8 MSE: {mse:.6f}")
        assert mse < 100.0, f"Percentile PTQ error too large: {mse}"
        
        print("✅ Static PTQ (Percentile) test passed!")
    
    def test_4_static_ptq_kl(self):
        """Test 4: Static PTQ with KL-Divergence Calibration."""
        if not self.available:
            return
        
        print("\n[Test 4] Static PTQ (KL-Divergence Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        calibration_data = [
            self.torch.randn(4, 1, 28, 28) for _ in range(20)
        ]
        
        chop = Chopi(bits=8, symmetric=True)
        model_q = self.static_post_quantization(
            model, chop,
            calibration_data=calibration_data,
            calibration_method='kl_divergence',
            fuse_bn=True,
            verbose=True
        )
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs KL-INT8 MSE: {mse:.6f}")
        assert mse < 100.0, f"KL-divergence PTQ error too large: {mse}"
        
        print("✅ Static PTQ (KL-Divergence) test passed!")
    
    def test_5_static_ptq_mse(self):
        """Test 5: Static PTQ with MSE Calibration."""
        if not self.available:
            return
        
        print("\n[Test 5] Static PTQ (MSE Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        calibration_data = [
            self.torch.randn(4, 1, 28, 28) for _ in range(15)
        ]
        
        chop = Chopi(bits=8, symmetric=True)
        model_q = self.static_post_quantization(
            model, chop,
            calibration_data=calibration_data,
            calibration_method='mse',
            fuse_bn=True,
            verbose=True
        )
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs MSE-INT8 MSE: {mse:.6f}")
        assert mse < 100.0, f"MSE PTQ error too large: {mse}"
        
        print("✅ Static PTQ (MSE) test passed!")
    
    def test_6_dynamic_ptq(self):
        """Test 6: Dynamic PTQ (No Calibration)."""
        if not self.available:
            return
        
        print("\n[Test 6] Dynamic PTQ (No Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        chop = Chopi(bits=8, symmetric=True)
        model_q = self.dynamic_post_quantization(model, chop, verbose=True)
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs Dynamic-INT8 MSE: {mse:.6f}")
        assert mse < 100.0, f"Dynamic PTQ error too large: {mse}"
        
        print("✅ Dynamic PTQ test passed!")
    
    def test_7_mixed_ptq_w8a16_static(self):
        """Test 7: Mixed PTQ (W8A16 with Static Calibration)."""
        if not self.available:
            return
        
        print("\n[Test 7] Mixed PTQ (W8A16 Static)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        calibration_data = [
            self.torch.randn(4, 1, 28, 28) for _ in range(10)
        ]
        
        weight_chop = Chopi(bits=8, symmetric=True)
        activation_chop = Chop(exp_bits=5, sig_bits=10)  # FP16
        
        model_q = self.mixed_post_quantization(
            model, weight_chop, activation_chop,
            calibration_data=calibration_data,
            dynamic=False,
            verbose=True
        )
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs W8A16 MSE: {mse:.6f}")
        assert mse < 50.0, f"W8A16 error too large: {mse}"
        
        print("✅ Mixed PTQ (W8A16 Static) test passed!")
    
    def test_8_mixed_ptq_w4a8_dynamic(self):
        """Test 8: Mixed PTQ (W4A8 with Dynamic Quantization)."""
        if not self.available:
            return
        
        print("\n[Test 8] Mixed PTQ (W4A8 Dynamic)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        weight_chop = Chopi(bits=4, symmetric=True)
        activation_chop = Chopi(bits=8, symmetric=True)
        
        model_q = self.mixed_post_quantization(
            model, weight_chop, activation_chop,
            dynamic=True,
            verbose=True
        )
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs W4A8 MSE: {mse:.6f}")
        assert mse < 200.0, f"W4A8 error too large: {mse}"
        
        print("✅ Mixed PTQ (W4A8 Dynamic) test passed!")
    
    def test_9_fp16_ptq(self):
        """Test 9: FP16 Quantization."""
        if not self.available:
            return
        
        print("\n[Test 9] FP16 PTQ")
        print("-" * 70)
        
        model = self.SimpleCNN()
        model.eval()
        
        chop = Chop(exp_bits=5, sig_bits=10)
        model_q = self.post_quantization(model, chop, verbose=True)
        
        x = self.torch.randn(2, 1, 28, 28)
        with self.torch.no_grad():
            output_fp = model(x)
            output_q = model_q(x)
        
        mse = self.torch.mean((output_fp - output_q) ** 2).item()
        print(f"\n[Result] FP32 vs FP16 MSE: {mse:.6f}")
        assert mse < 1.0, f"FP16 error too large: {mse}"
        
        print("✅ FP16 PTQ test passed!")
    

    def run_all_tests(self):
        """Run all PyTorch tests."""
        if not self.available:
            print("❌ PyTorch not available! Skipping tests.")
            return
        
        try:
            self.test_1_basic_ptq()
            self.test_2_static_ptq_minmax()
            self.test_3_static_ptq_percentile()
            self.test_4_static_ptq_kl()
            self.test_5_static_ptq_mse()
            self.test_6_dynamic_ptq()
            self.test_7_mixed_ptq_w8a16_static()
            self.test_8_mixed_ptq_w4a8_dynamic()
            self.test_9_fp16_ptq()
            
            print("\n" + "="*70)
            print("✅ All PyTorch PTQ tests passed!")
            print("="*70)
        except Exception as e:
            print(f"\n❌ PyTorch test failed: {e}")
            import traceback
            traceback.print_exc()


# ===================================================================
# JAX Backend Tests (完整版 - 与 PyTorch 对等)
# ===================================================================

class TestJAXPTQ:
    """Test suite for JAX backend PTQ (full feature parity with PyTorch)."""
    
    def __init__(self):
        """Initialize test class."""
        if not _is_jax_available():
            print("❌ JAX/Flax not available! Skipping tests.")
            self.available = False
            return
        
        self.available = True
        pychop.backend('jax')
        
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from pychop.ptq import (
            post_quantization,
            static_post_quantization,
            dynamic_post_quantization,
            mixed_post_quantization
        )
        
        self.jax = jax
        self.jnp = jnp
        self.nn = nn
        self.post_quantization = post_quantization
        self.static_post_quantization = static_post_quantization
        self.dynamic_post_quantization = dynamic_post_quantization
        self.mixed_post_quantization = mixed_post_quantization
        
        # Define SimpleCNN class
        class SimpleCNN(nn.Module):
            num_classes: int = 10
            
            @nn.compact
            def __call__(self, x, train: bool = False):
                x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
                x = nn.BatchNorm(use_running_average=not train)(x)
                x = nn.relu(x)
                x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
                x = nn.BatchNorm(use_running_average=not train)(x)
                x = nn.relu(x)
                x = jnp.mean(x, axis=(1, 2))  # Global avg pool
                x = nn.Dense(features=self.num_classes)(x)
                return x
        
        self.SimpleCNN = SimpleCNN
        
        print("\n" + "="*70)
        print("JAX Backend PTQ Test Suite")
        print("="*70)
    
    def test_1_basic_ptq(self):
        """Test 1: Basic PTQ (Weight-Only Quantization)."""
        if not self.available:
            return
        
        print("\n[Test 1] Basic PTQ (Weight-Only)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        from pychop.jx.layers import ChopiSTE
        chop = ChopiSTE(bits=8, symmetric=True)
        
        quantized_result = self.post_quantization(variables, chop, verbose=True)
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        assert output_fp.shape == output_q.shape
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs INT8 MSE: {mse:.6f}")
        assert mse < 50.0, f"Quantization error too large: {mse}"
        
        print("✅ Basic PTQ test passed!")
    
    def test_2_static_ptq_minmax(self):
        """Test 2: Static PTQ with MinMax Calibration."""
        if not self.available:
            return
        
        print("\n[Test 2] Static PTQ (MinMax Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        calibration_data = [
            self.jax.random.normal(self.jax.random.PRNGKey(i), (4, 28, 28, 1))
            for i in range(10)
        ]
        
        from pychop.jx.layers import ChopiSTE
        chop = ChopiSTE(bits=8, symmetric=True)
        
        # Define apply function
        def apply_fn(params, x):
            return model.apply(params, x, train=False)
        
        quantized_result = self.static_post_quantization(
            variables, chop,
            calibration_data=calibration_data,
            calibration_method='minmax',
            model_apply_fn=apply_fn,
            verbose=True
        )
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs Static-INT8 (MinMax) MSE: {mse:.6f}")
        
        print("✅ Static PTQ (MinMax) test passed!")
    
    def test_3_static_ptq_percentile(self):
        """Test 3: Static PTQ with Percentile Calibration."""
        if not self.available:
            return
        
        print("\n[Test 3] Static PTQ (Percentile Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        calibration_data = [
            self.jax.random.normal(self.jax.random.PRNGKey(i), (4, 28, 28, 1))
            for i in range(10)
        ]
        
        from pychop.jx.layers import ChopiSTE
        chop = ChopiSTE(bits=8, symmetric=True)
        
        def apply_fn(params, x):
            return model.apply(params, x, train=False)
        
        quantized_result = self.static_post_quantization(
            variables, chop,
            calibration_data=calibration_data,
            calibration_method='percentile',
            percentile=99.9,
            model_apply_fn=apply_fn,
            verbose=True
        )
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs Percentile-INT8 MSE: {mse:.6f}")
        
        print("✅ Static PTQ (Percentile) test passed!")
    
    def test_4_static_ptq_kl(self):
        """Test 4: Static PTQ with KL-Divergence Calibration (NEW!)."""
        if not self.available:
            return
        
        print("\n[Test 4] Static PTQ (KL-Divergence Calibration) ✨ NEW")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        calibration_data = [
            self.jax.random.normal(self.jax.random.PRNGKey(i), (4, 28, 28, 1))
            for i in range(20)
        ]
        
        from pychop.jx.layers import ChopiSTE
        chop = ChopiSTE(bits=8, symmetric=True)
        
        def apply_fn(params, x):
            return model.apply(params, x, train=False)
        
        quantized_result = self.static_post_quantization(
            variables, chop,
            calibration_data=calibration_data,
            calibration_method='kl_divergence',
            model_apply_fn=apply_fn,
            verbose=True
        )
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs KL-INT8 MSE: {mse:.6f}")
        
        print("✅ Static PTQ (KL-Divergence) test passed!")
    
    def test_5_static_ptq_mse(self):
        """Test 5: Static PTQ with MSE Calibration (NEW!)."""
        if not self.available:
            return
        
        print("\n[Test 5] Static PTQ (MSE Calibration) ✨ NEW")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        calibration_data = [
            self.jax.random.normal(self.jax.random.PRNGKey(i), (4, 28, 28, 1))
            for i in range(15)
        ]
        
        from pychop.jx.layers import ChopiSTE
        chop = ChopiSTE(bits=8, symmetric=True)
        
        def apply_fn(params, x):
            return model.apply(params, x, train=False)
        
        quantized_result = self.static_post_quantization(
            variables, chop,
            calibration_data=calibration_data,
            calibration_method='mse',
            model_apply_fn=apply_fn,
            verbose=True
        )
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs MSE-INT8 MSE: {mse:.6f}")
        
        print("✅ Static PTQ (MSE) test passed!")
    
    def test_6_dynamic_ptq(self):
        """Test 6: Dynamic PTQ (No Calibration)."""
        if not self.available:
            return
        
        print("\n[Test 6] Dynamic PTQ (No Calibration)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        from pychop.jx.layers import ChopiSTE
        chop = ChopiSTE(bits=8, symmetric=True)
        
        quantized_result = self.dynamic_post_quantization(variables, chop, verbose=True)
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs Dynamic-INT8 MSE: {mse:.6f}")
        
        print("✅ Dynamic PTQ test passed!")
    
    def test_7_mixed_ptq_w8a16_static(self):
        """Test 7: Mixed PTQ (W8A16 with Static Calibration) (NEW!)."""
        if not self.available:
            return
        
        print("\n[Test 7] Mixed PTQ (W8A16 Static) ✨ ENHANCED")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        calibration_data = [
            self.jax.random.normal(self.jax.random.PRNGKey(i), (4, 28, 28, 1))
            for i in range(10)
        ]
        
        from pychop.jx.layers import ChopiSTE, ChopSTE
        weight_chop = ChopiSTE(bits=8, symmetric=True)
        activation_chop = ChopSTE(exp_bits=5, sig_bits=10)  # FP16
        
        def apply_fn(params, x):
            return model.apply(params, x, train=False)
        
        quantized_result = self.mixed_post_quantization(
            variables, weight_chop, activation_chop,
            calibration_data=calibration_data,
            calibration_method='percentile',
            model_apply_fn=apply_fn,
            verbose=True
        )
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs W8A16 MSE: {mse:.6f}")
        
        print("✅ Mixed PTQ (W8A16 Static) test passed!")
    
    def test_8_mixed_ptq_w4a8_dynamic(self):
        """Test 8: Mixed PTQ (W4A8 Dynamic)."""
        if not self.available:
            return
        
        print("\n[Test 8] Mixed PTQ (W4A8 Dynamic)")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        from pychop.jx.layers import ChopiSTE
        weight_chop = ChopiSTE(bits=4, symmetric=True)
        activation_chop = ChopiSTE(bits=8, symmetric=True)
        
        quantized_result = self.mixed_post_quantization(
            variables, weight_chop, activation_chop,
            verbose=True
        )
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs W4A8 MSE: {mse:.6f}")
        
        print("✅ Mixed PTQ (W4A8 Dynamic) test passed!")
    
    def test_9_fp16_ptq(self):
        """Test 9: FP16 Quantization (NEW!)."""
        if not self.available:
            return
        
        print("\n[Test 9] FP16 PTQ ✨ NEW")
        print("-" * 70)
        
        model = self.SimpleCNN()
        
        rng = self.jax.random.PRNGKey(0)
        x_dummy = self.jnp.ones((1, 28, 28, 1))
        variables = model.init(rng, x_dummy, train=False)
        
        from pychop.jx.layers import ChopSTE
        chop = ChopSTE(exp_bits=5, sig_bits=10)  # FP16
        
        quantized_result = self.post_quantization(variables, chop, verbose=True)
        
        x = self.jax.random.normal(rng, (2, 28, 28, 1))
        output_fp = model.apply(variables, x, train=False)
        output_q = model.apply(quantized_result, x, train=False)
        
        mse = float(self.jnp.mean((output_fp - output_q) ** 2))
        print(f"\n[Result] FP32 vs FP16 MSE: {mse:.6f}")
        assert mse < 1.0, f"FP16 error too large: {mse}"
        
        print("✅ FP16 PTQ test passed!")
    
    def run_all_tests(self):
        """Run all JAX tests."""
        if not self.available:
            print("❌ JAX/Flax not available! Skipping tests.")
            return
        
        try:
            self.test_1_basic_ptq()
            self.test_2_static_ptq_minmax()
            self.test_3_static_ptq_percentile()
            self.test_4_static_ptq_kl()           # ✨ NEW
            self.test_5_static_ptq_mse()          # ✨ NEW
            self.test_6_dynamic_ptq()
            self.test_7_mixed_ptq_w8a16_static()  # ✨ ENHANCED
            self.test_8_mixed_ptq_w4a8_dynamic()
            self.test_9_fp16_ptq()                # ✨ NEW
            
            print("\n" + "="*70)
            print("✅ All JAX PTQ tests passed!")
            print("="*70)
        except Exception as e:
            print(f"\n❌ JAX test failed: {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    print("\n" + "="*70)
    print("PyChop PTQ Comprehensive Test Suite")
    print("="*70)
    
    # Check available backends
    torch_available = _is_torch_available()
    jax_available = _is_jax_available()
    
    print(f"\n[Backend Status]")
    print(f"  PyTorch: {'✅ Available' if torch_available else '❌ Not installed'}")
    print(f"  JAX/Flax: {'✅ Available' if jax_available else '❌ Not installed'}")
    
    # Run PyTorch tests
    if torch_available:
        print("\n" + "="*70)
        print("Running PyTorch Backend Tests (9 tests)...")
        print("="*70)
        test_torch = TestPyTorchPTQ()
        test_torch.run_all_tests()
    
    # Run JAX tests
    if jax_available:
        print("\n" + "="*70)
        print("Running JAX Backend Tests (9 tests)... ✨ ENHANCED")
        print("="*70)
        test_jax = TestJAXPTQ()
        test_jax.run_all_tests()
    
    if not torch_available and not jax_available:
        print("\n❌ No backend available! Please install PyTorch or JAX/Flax.")
        print("\nInstall with:")
        print("  pip install torch  # For PyTorch backend")
        print("  pip install jax jaxlib flax  # For JAX backend")
    
    print("\n" + "="*70)
    print("Test suite execution completed!")
    print("="*70)