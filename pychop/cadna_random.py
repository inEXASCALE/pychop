"""
CADNA-style random number generator for sign-flip stochastic rounding.

Based on the original CADNA implementation using Tausworthe + LCG generators.
Supports NumPy, PyTorch, and JAX backends.
"""

import numpy as np
from typing import Optional, Union


class CADNARandomGenerator:
    """
    High-performance random bit generator mimicking CADNA's implementation.
    
    Uses 4 independent generators:
    - 3x Tausworthe generators
    - 1x Linear Congruential Generator (LCG)
    
    Results are XORed together and cached for 32 operations.
    """
    
    # Tausworthe parameters (from CADNA source)
    TAUS1_S1, TAUS1_S2, TAUS1_S3, TAUS1_M = 13, 19, 12, 4294967294
    TAUS2_S1, TAUS2_S2, TAUS2_S3, TAUS2_M = 2, 25, 4, 4294967288
    TAUS3_S1, TAUS3_S2, TAUS3_S3, TAUS3_M = 3, 11, 17, 4294967280
    
    # LCG parameters
    LCG_A, LCG_C = 1664525, 1013904223
    
    def __init__(self, seed: int = 0, backend: str = "numpy"):
        """
        Initialize CADNA random generator.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        backend : str
            Backend type: "numpy", "torch", or "jax"
        """
        self.backend = backend
        self._init_seeds(seed)
        self._random_cache = np.uint32(0)
        self._cache_counter = 0
        
    def _init_seeds(self, seed: int):
        """Initialize generator seeds."""
        np.random.seed(seed)
        
        # Initialize Tausworthe seeds (must satisfy bit constraints)
        self.z1 = np.uint32(np.random.randint(128, 2**32 - 1))
        self.z2 = np.uint32(np.random.randint(128, 2**32 - 1))
        self.z3 = np.uint32(np.random.randint(128, 2**32 - 1))
        self.z4 = np.uint32(np.random.randint(1, 2**32 - 1))
        
    def _tausworthe_step(self, z: np.uint32, S1: int, S2: int, S3: int, M: np.uint32) -> np.uint32:
        """Single Tausworthe generator step."""
        z = np.uint32(z)
        M = np.uint32(M)
        b = np.uint32(((z << np.uint32(S1)) ^ z) >> np.uint32(S2))
        z = np.uint32(((z & M) << np.uint32(S3)) ^ b)
        return z
        
    def _lcg_step(self) -> np.uint32:
        """Linear Congruential Generator step."""
        self.z4 = np.uint32((np.uint64(self.LCG_A) * np.uint64(self.z4) + np.uint64(self.LCG_C)) % (2**32))
        return self.z4
        
    def _generate_batch(self) -> np.uint32:
        """Generate 32 random bits at once."""
        self.z1 = self._tausworthe_step(self.z1, self.TAUS1_S1, self.TAUS1_S2, 
                                         self.TAUS1_S3, np.uint32(self.TAUS1_M))
        self.z2 = self._tausworthe_step(self.z2, self.TAUS2_S1, self.TAUS2_S2, 
                                         self.TAUS2_S3, np.uint32(self.TAUS2_M))
        self.z3 = self._tausworthe_step(self.z3, self.TAUS3_S1, self.TAUS3_S2, 
                                         self.TAUS3_S3, np.uint32(self.TAUS3_M))
        lcg_val = self._lcg_step()
        
        # XOR all generators together
        return np.uint32(self.z1 ^ self.z2 ^ self.z3 ^ lcg_val)
        
    def random_bit(self) -> int:
        """
        Generate a single random bit (0 or 1).
        Uses caching: generates 32 bits at once, returns one at a time.
        """
        if self._cache_counter % 32 == 0:
            self._random_cache = self._generate_batch()
            self._cache_counter = 0
            
        bit = int(self._random_cache & np.uint32(1))
        self._random_cache = np.uint32(self._random_cache >> np.uint32(1))
        self._cache_counter += 1
        
        return bit
        
    def random_bits(self, shape: tuple) -> np.ndarray:
        """
        Generate an array of random bits.
        
        Parameters
        ----------
        shape : tuple
            Shape of output array
            
        Returns
        -------
        np.ndarray
            Array of random bits (0 or 1), dtype=uint8
        """
        size = int(np.prod(shape))
        bits = np.array([self.random_bit() for _ in range(size)], dtype=np.uint8)
        return bits.reshape(shape)


# Backend-specific bit flip implementations

def numpy_bit_flip(x: np.ndarray, random_bits: np.ndarray) -> np.ndarray:
    """
    Flip sign bit using NumPy.
    
    When random_bit=1, flip the sign; when random_bit=0, keep original sign.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    random_bits : np.ndarray
        Random bits array (0 or 1), same shape as x
        
    Returns
    -------
    np.ndarray
        Array with sign bits flipped according to random_bits
    """
    # View as unsigned integer to manipulate bits
    if x.dtype == np.float64:
        x_int = x.view(np.uint64)
        # Create sign bit mask using Python int to avoid overflow
        sign_bit = np.uint64(1 << 63)
        # Apply XOR: when random_bit=1, flip sign; when random_bit=0, keep sign
        mask = random_bits.astype(np.uint64) * sign_bit
        x_int_flipped = x_int ^ mask
        return x_int_flipped.view(np.float64)
        
    elif x.dtype == np.float32:
        x_int = x.view(np.uint32)
        sign_bit = np.uint32(1 << 31)
        mask = random_bits.astype(np.uint32) * sign_bit
        x_int_flipped = x_int ^ mask
        return x_int_flipped.view(np.float32)
        
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def torch_bit_flip(x, random_bits):
    """
    Flip sign bit using PyTorch.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    random_bits : torch.Tensor
        Random bits tensor (0 or 1), same shape as x
        
    Returns
    -------
    torch.Tensor
        Tensor with sign bits flipped according to random_bits
    """
    import torch
    
    # View as integer to manipulate bits
    if x.dtype == torch.float64:
        x_int = x.view(torch.int64)
        sign_bit = torch.tensor(1 << 63, dtype=torch.int64, device=x.device)
        mask = random_bits.to(torch.int64) * sign_bit
        x_int_flipped = x_int ^ mask
        return x_int_flipped.view(torch.float64)
        
    elif x.dtype == torch.float32:
        x_int = x.view(torch.int32)
        sign_bit = torch.tensor(1 << 31, dtype=torch.int32, device=x.device)
        mask = random_bits.to(torch.int32) * sign_bit
        x_int_flipped = x_int ^ mask
        return x_int_flipped.view(torch.float32)
        
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def jax_bit_flip(x, random_bits):
    """
    Flip sign bit using JAX.
    
    Parameters
    ----------
    x : jax.numpy.ndarray
        Input array
    random_bits : jax.numpy.ndarray
        Random bits array (0 or 1), same shape as x
        
    Returns
    -------
    jax.numpy.ndarray
        Array with sign bits flipped according to random_bits
    """
    import jax.numpy as jnp
    
    # View as integer to manipulate bits
    if x.dtype == jnp.float64:
        x_int = x.view(jnp.uint64)
        sign_bit = jnp.uint64(1 << 63)
        mask = random_bits.astype(jnp.uint64) * sign_bit
        x_int_flipped = x_int ^ mask
        return x_int_flipped.view(jnp.float64)
        
    elif x.dtype == jnp.float32:
        x_int = x.view(jnp.uint32)
        sign_bit = jnp.uint32(1 << 31)
        mask = random_bits.astype(jnp.uint32) * sign_bit
        x_int_flipped = x_int ^ mask
        return x_int_flipped.view(jnp.float32)
        
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def cadna_round(x, backend: str = "numpy", random_gen: Optional[CADNARandomGenerator] = None):
    """
    Apply CADNA-style random rounding via sign-flip.
    
    This function performs: x' = (-1)^r * x where r ~ Bernoulli(0.5)
    When applied before and after an operation, it simulates stochastic rounding.
    
    Parameters
    ----------
    x : array-like
        Input array (NumPy, PyTorch, or JAX)
    backend : str
        Backend type: "numpy", "torch", or "jax"
    random_gen : CADNARandomGenerator, optional
        Random generator instance. If None, creates a new one.
        
    Returns
    -------
    array-like
        Array with signs flipped according to random bits
    """
    if random_gen is None:
        random_gen = CADNARandomGenerator(backend=backend)
        
    # Generate random bits
    shape = x.shape
    random_bits = random_gen.random_bits(shape)
    
    # Convert to appropriate backend
    if backend == "numpy":
        import numpy as np
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return numpy_bit_flip(x.copy(), random_bits)
        
    elif backend == "torch":
        import torch
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        random_bits_torch = torch.from_numpy(random_bits).to(x.device)
        return torch_bit_flip(x.clone(), random_bits_torch)
        
    elif backend == "jax":
        import jax.numpy as jnp
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)
        random_bits_jax = jnp.array(random_bits)
        return jax_bit_flip(x, random_bits_jax)
        
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# Test function
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("CADNA Random Generator Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Random generator
    print("Test 1: Random Bit Generator")
    print("-" * 40)
    gen = CADNARandomGenerator(seed=42)
    bits = [gen.random_bit() for _ in range(1000)]
    mean_val = np.mean(bits)
    print(f"Generated 1000 bits")
    print(f"Mean: {mean_val:.4f} (expected: ~0.5)")
    print(f"Min: {min(bits)}, Max: {max(bits)}")
    assert 0.45 < mean_val < 0.55, "Mean should be close to 0.5"
    print("✓ Test passed\n")
    
    # Test 2: NumPy bit flip logic
    print("Test 2: NumPy Bit Flip Logic")
    print("-" * 40)
    x_np = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float64)
    random_bits = np.array([0, 0, 1, 1], dtype=np.uint8)  # Fixed pattern for testing
    x_flipped = numpy_bit_flip(x_np.copy(), random_bits)
    
    # When random_bit=1, sign should flip; when random_bit=0, sign stays same
    # Expected: [1.5, -2.3, -3.7, 0.5]
    #           (no flip, no flip, flip, flip)
    expected = np.array([1.5, -2.3, -3.7, 0.5])
    
    print(f"Original:     {x_np}")
    print(f"Random bits:  {random_bits} (1=flip, 0=keep)")
    print(f"Flipped:      {x_flipped}")
    print(f"Expected:     {expected}")
    print(f"Match: {np.allclose(x_flipped, expected)}")
    
    assert np.allclose(x_flipped, expected), f"Expected {expected}, got {x_flipped}"
    print("✓ Test passed\n")
    
    # Test 3: Sign preservation check
    print("Test 3: Sign Preservation Logic")
    print("-" * 40)
    x_test = np.array([2.0, -3.0, 4.0, -5.0])
    
    # Test with all zeros (no flip)
    bits_zero = np.zeros(4, dtype=np.uint8)
    result_zero = numpy_bit_flip(x_test.copy(), bits_zero)
    print(f"No flip (bits=0): {x_test} -> {result_zero}")
    assert np.array_equal(x_test, result_zero), "With bits=0, should be identical"
    
    # Test with all ones (all flip)
    bits_one = np.ones(4, dtype=np.uint8)
    result_one = numpy_bit_flip(x_test.copy(), bits_one)
    print(f"All flip (bits=1): {x_test} -> {result_one}")
    assert np.array_equal(result_one, -x_test), "With bits=1, should negate all"
    print("✓ Test passed\n")
    
    # Test 4: Float32
    print("Test 4: Float32 Bit Flip")
    print("-" * 40)
    x_f32 = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float32)
    random_bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    x_flipped_f32 = numpy_bit_flip(x_f32.copy(), random_bits)
    
    expected_f32 = np.array([-1.5, -2.3, -3.7, -0.5], dtype=np.float32)
    
    print(f"Original:     {x_f32}")
    print(f"Random bits:  {random_bits}")
    print(f"Flipped:      {x_flipped_f32}")
    print(f"Expected:     {expected_f32}")
    print(f"Match: {np.allclose(x_flipped_f32, expected_f32)}")
    
    assert np.allclose(x_flipped_f32, expected_f32), "Float32 flip failed"
    print("✓ Test passed\n")
    
    # Test 5: PyTorch (if available)
    print("Test 5: PyTorch Backend")
    print("-" * 40)
    try:
        import torch
        x_torch = torch.tensor([1.5, -2.3, 3.7, -0.5])
        random_bits_torch = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
        x_flipped_torch = torch_bit_flip(x_torch.clone(), random_bits_torch)
        
        expected_torch = torch.tensor([1.5, -2.3, -3.7, 0.5])
        
        print(f"Original: {x_torch}")
        print(f"Flipped:  {x_flipped_torch}")
        print(f"Expected: {expected_torch}")
        print(f"Match: {torch.allclose(x_flipped_torch, expected_torch)}")
        
        assert torch.allclose(x_flipped_torch, expected_torch), "PyTorch flip failed"
        print("✓ Test passed\n")
    except ImportError:
        print("⊘ PyTorch not available, skipping\n")
    
    # Test 6: JAX (if available)
    print("Test 6: JAX Backend")
    print("-" * 40)
    try:
        import jax.numpy as jnp
        x_jax = jnp.array([1.5, -2.3, 3.7, -0.5])
        random_bits_jax = jnp.array([0, 0, 1, 1], dtype=jnp.uint8)
        x_flipped_jax = jax_bit_flip(x_jax, random_bits_jax)
        
        expected_jax = jnp.array([1.5, -2.3, -3.7, 0.5])
        
        print(f"Original: {x_jax}")
        print(f"Flipped:  {x_flipped_jax}")
        print(f"Expected: {expected_jax}")
        print(f"Match: {jnp.allclose(x_flipped_jax, expected_jax)}")
        
        assert jnp.allclose(x_flipped_jax, expected_jax), "JAX flip failed"
        print("✓ Test passed\n")
    except ImportError:
        print("⊘ JAX not available, skipping\n")
    
    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)