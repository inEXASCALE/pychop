"""
CADNA-style random bit generator and sign-bit flip utilities.

Important:
    A sign flip alone is NOT random rounding.

    CADNA-style stochastic rounding for arithmetic operations relies on this idea:

        round_down(a op b) == -round_up((-a) op (-b))

    Therefore, sign-bit flipping is useful when it is applied before and after
    an arithmetic operation that is evaluated under a directed rounding mode,
    typically FE_UPWARD in CADNA-like implementations.

    This module provides:
        - a CADNA-style random bit generator
        - NumPy / PyTorch / JAX sign-bit flip helpers
        - cadna_round(), kept for compatibility with the original code, although
          it is more accurately a random sign-flip helper than a full rounding function.

Supports NumPy, PyTorch, and JAX backends.
"""

from typing import Optional
import numpy as np


class CADNARandomGenerator:
    """
    CADNA-style random bit generator for sign-flip stochastic rounding.

    Uses a combined Tausworthe + LCG generator and caches 32 bits at a time.

    Notes
    -----
    This is CADNA-style, not a guaranteed bit-for-bit clone of every CADNA source
    distribution. To reproduce a specific CADNA implementation exactly, its seed
    initialization and internal state update logic must be copied exactly.
    """

    # Standard combined Tausworthe masks commonly used with this generator family.
    # These are the values used in the user's original implementation.
    TAUS1_S1, TAUS1_S2, TAUS1_S3, TAUS1_M = 13, 19, 12, 4294967294
    TAUS2_S1, TAUS2_S2, TAUS2_S3, TAUS2_M = 2, 25, 4, 4294967288
    TAUS3_S1, TAUS3_S2, TAUS3_S3, TAUS3_M = 3, 11, 17, 4294967280

    # LCG parameters
    LCG_A, LCG_C = 1664525, 1013904223

    UINT32_MASK = np.uint64(0xFFFFFFFF)

    def __init__(self, seed: int = 0, backend: str = "numpy"):
        """
        Initialize CADNA-style random generator.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        backend : str
            Backend type: "numpy", "torch", or "jax".
            Stored for compatibility; bit generation itself is NumPy-based.
        """
        self.backend = backend
        self._init_seeds(seed)
        self._random_cache = np.uint32(0)
        self._cache_counter = 32  # force generation on first random_bit()

    def _init_seeds(self, seed: int):
        """
        Initialize generator seeds without modifying NumPy's global RNG state.
        """
        rng = np.random.default_rng(seed)

        # Tausworthe generators require non-trivial nonzero seeds.
        # These lower bounds are conservative and avoid pathological small states.
        self.z1 = np.uint32(rng.integers(128, 2**32 - 1, dtype=np.uint32))
        self.z2 = np.uint32(rng.integers(128, 2**32 - 1, dtype=np.uint32))
        self.z3 = np.uint32(rng.integers(128, 2**32 - 1, dtype=np.uint32))
        self.z4 = np.uint32(rng.integers(1, 2**32 - 1, dtype=np.uint32))

    @staticmethod
    def _uint32(x) -> np.uint32:
        """
        Convert to uint32 with explicit modulo 2**32 semantics.
        """
        return np.uint32(np.uint64(x) & CADNARandomGenerator.UINT32_MASK)

    def _tausworthe_step(
        self,
        z: np.uint32,
        S1: int,
        S2: int,
        S3: int,
        M: int,
    ) -> np.uint32:
        """
        Single Tausworthe generator step with explicit uint32 wraparound.
        """
        z64 = np.uint64(z)
        m64 = np.uint64(M)

        b = ((z64 << np.uint64(S1)) ^ z64) >> np.uint64(S2)
        z_new = (((z64 & m64) << np.uint64(S3)) ^ b) & self.UINT32_MASK

        return np.uint32(z_new)

    def _lcg_step(self) -> np.uint32:
        """
        Linear Congruential Generator step with uint32 wraparound.
        """
        z = (
            np.uint64(self.LCG_A) * np.uint64(self.z4)
            + np.uint64(self.LCG_C)
        ) & self.UINT32_MASK

        self.z4 = np.uint32(z)
        return self.z4

    def _generate_batch(self) -> np.uint32:
        """
        Generate 32 random bits at once.
        """
        self.z1 = self._tausworthe_step(
            self.z1,
            self.TAUS1_S1,
            self.TAUS1_S2,
            self.TAUS1_S3,
            self.TAUS1_M,
        )
        self.z2 = self._tausworthe_step(
            self.z2,
            self.TAUS2_S1,
            self.TAUS2_S2,
            self.TAUS2_S3,
            self.TAUS2_M,
        )
        self.z3 = self._tausworthe_step(
            self.z3,
            self.TAUS3_S1,
            self.TAUS3_S2,
            self.TAUS3_S3,
            self.TAUS3_M,
        )
        z4 = self._lcg_step()

        return np.uint32(self.z1 ^ self.z2 ^ self.z3 ^ z4)

    def random_bit(self) -> int:
        """
        Generate a single random bit, 0 or 1.

        Uses caching: generates 32 bits at once and consumes one bit per call.
        """
        if self._cache_counter >= 32:
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
            Shape of output array.

        Returns
        -------
        np.ndarray
            Array of random bits, values 0 or 1, dtype uint8.
        """
        if shape == ():
            return np.array(self.random_bit(), dtype=np.uint8)

        size = int(np.prod(shape))
        bits = np.fromiter(
            (self.random_bit() for _ in range(size)),
            dtype=np.uint8,
            count=size,
        )
        return bits.reshape(shape)


def numpy_bit_flip(x: np.ndarray, random_bits: np.ndarray) -> np.ndarray:
    """
    Flip sign bits using NumPy.

    When random_bit=1, flip the sign bit.
    When random_bit=0, keep the original sign bit.

    Parameters
    ----------
    x : np.ndarray
        Input array with dtype float32 or float64.
    random_bits : np.ndarray
        Random bits array, broadcastable to x.shape.

    Returns
    -------
    np.ndarray
        Array with sign bits flipped according to random_bits.
    """
    x = np.asarray(x)

    if x.dtype not in (np.float32, np.float64):
        raise ValueError(f"Unsupported dtype: {x.dtype}. Expected float32 or float64.")

    # Make a copy so the caller's input is not mutated.
    x_work = np.array(x, copy=True)

    random_bits = np.asarray(random_bits, dtype=np.uint8)
    random_bits = np.broadcast_to(random_bits, x_work.shape)

    if x_work.dtype == np.float64:
        x_int = x_work.view(np.uint64)
        sign_bit = np.uint64(1 << 63)
        mask = random_bits.astype(np.uint64) * sign_bit
        return (x_int ^ mask).view(np.float64)

    # float32
    x_int = x_work.view(np.uint32)
    sign_bit = np.uint32(1 << 31)
    mask = random_bits.astype(np.uint32) * sign_bit
    return (x_int ^ mask).view(np.float32)


def torch_bit_flip(x, random_bits):
    """
    Flip sign bits using PyTorch.

    This implementation deliberately uses torch.where(..., -x, x) instead of
    integer sign masks, because constructing 1 << 63 as torch.int64 overflows.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with dtype torch.float32 or torch.float64.
    random_bits : torch.Tensor
        Random bits tensor, broadcastable to x.shape.

    Returns
    -------
    torch.Tensor
        Tensor with signs flipped according to random_bits.
    """
    import torch

    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Unsupported dtype: {x.dtype}. Expected float32 or float64.")

    if not isinstance(random_bits, torch.Tensor):
        random_bits = torch.as_tensor(random_bits, device=x.device)

    random_bits = random_bits.to(device=x.device)
    flip_mask = random_bits.to(torch.bool)

    return torch.where(flip_mask, -x, x)


def jax_bit_flip(x, random_bits):
    """
    Flip sign bits using JAX.

    This implementation uses jnp.where(..., -x, x), which is the stable JAX
    expression for conditional sign flipping across devices.

    Parameters
    ----------
    x : jax.numpy.ndarray
        Input array with dtype float32 or float64.
    random_bits : jax.numpy.ndarray
        Random bits array, broadcastable to x.shape.

    Returns
    -------
    jax.numpy.ndarray
        Array with signs flipped according to random_bits.
    """
    import jax.numpy as jnp

    x = jnp.asarray(x)

    if x.dtype not in (jnp.float32, jnp.float64):
        raise ValueError(f"Unsupported dtype: {x.dtype}. Expected float32 or float64.")

    random_bits = jnp.asarray(random_bits)
    flip_mask = random_bits.astype(jnp.bool_)

    return jnp.where(flip_mask, -x, x)


def cadna_round(x, backend: str = "numpy", random_gen: Optional[CADNARandomGenerator] = None):
    """
    Apply CADNA-style random sign flip.

    Warning
    -------
    This function alone is not a complete random rounding operation.

    It performs:

        x' = x   if random_bit == 0
        x' = -x  if random_bit == 1

    CADNA-style stochastic rounding for an operation requires applying this
    sign flip consistently before and after the operation under a directed
    rounding environment.

    Parameters
    ----------
    x : array-like
        Input array or tensor.
    backend : str
        Backend type: "numpy", "torch", or "jax".
    random_gen : CADNARandomGenerator, optional
        Random generator instance. If None, creates a new one.

    Returns
    -------
    array-like
        Array or tensor with signs flipped according to random bits.
    """
    if random_gen is None:
        random_gen = CADNARandomGenerator(backend=backend)

    if backend == "numpy":
        x = np.asarray(x)

        if x.dtype not in (np.float32, np.float64):
            x = x.astype(np.float64)

        random_bits = random_gen.random_bits(x.shape)
        return numpy_bit_flip(x, random_bits)

    if backend == "torch":
        import torch

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)

        if x.dtype not in (torch.float32, torch.float64):
            x = x.to(torch.float64)

        random_bits = random_gen.random_bits(tuple(x.shape))
        random_bits_torch = torch.from_numpy(np.asarray(random_bits)).to(x.device)

        return torch_bit_flip(x, random_bits_torch)

    if backend == "jax":
        import jax.numpy as jnp

        x = jnp.asarray(x)

        if x.dtype not in (jnp.float32, jnp.float64):
            x = x.astype(jnp.float64)

        random_bits = random_gen.random_bits(tuple(x.shape))
        random_bits_jax = jnp.asarray(random_bits)

        return jax_bit_flip(x, random_bits_jax)

    raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    print("=" * 60)
    print("CADNA-style Random Generator Test Suite")
    print("=" * 60 + "\n")

    # Test 1: Random generator
    print("Test 1: Random Bit Generator")
    print("-" * 40)
    gen = CADNARandomGenerator(seed=42)
    bits = [gen.random_bit() for _ in range(1000)]
    mean_val = np.mean(bits)
    print("Generated 1000 bits")
    print(f"Mean: {mean_val:.4f} (expected: roughly 0.5)")
    print(f"Min: {min(bits)}, Max: {max(bits)}")
    assert 0.40 < mean_val < 0.60, "Mean should be reasonably close to 0.5"
    print("✓ Test passed\n")

    # Test 2: NumPy float64 bit flip logic
    print("Test 2: NumPy Float64 Bit Flip Logic")
    print("-" * 40)
    x_np = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float64)
    random_bits = np.array([0, 0, 1, 1], dtype=np.uint8)
    x_flipped = numpy_bit_flip(x_np, random_bits)

    expected = np.array([1.5, -2.3, -3.7, 0.5], dtype=np.float64)

    print(f"Original:     {x_np}")
    print(f"Random bits:  {random_bits} (1=flip, 0=keep)")
    print(f"Flipped:      {x_flipped}")
    print(f"Expected:     {expected}")
    print(f"Match:        {np.allclose(x_flipped, expected)}")

    assert np.allclose(x_flipped, expected), f"Expected {expected}, got {x_flipped}"
    print("✓ Test passed\n")

    # Test 3: Original array is not mutated
    print("Test 3: NumPy Input Is Not Mutated")
    print("-" * 40)
    x_original = np.array([2.0, -3.0, 4.0, -5.0], dtype=np.float64)
    x_before = x_original.copy()
    _ = numpy_bit_flip(x_original, np.ones_like(x_original, dtype=np.uint8))
    print(f"Before: {x_before}")
    print(f"After:  {x_original}")
    assert np.array_equal(x_original, x_before), "numpy_bit_flip should not mutate input"
    print("✓ Test passed\n")

    # Test 4: Sign preservation check
    print("Test 4: Sign Preservation Logic")
    print("-" * 40)
    x_test = np.array([2.0, -3.0, 4.0, -5.0], dtype=np.float64)

    bits_zero = np.zeros(4, dtype=np.uint8)
    result_zero = numpy_bit_flip(x_test, bits_zero)
    print(f"No flip:  {x_test} -> {result_zero}")
    assert np.array_equal(x_test, result_zero), "With bits=0, should be identical"

    bits_one = np.ones(4, dtype=np.uint8)
    result_one = numpy_bit_flip(x_test, bits_one)
    print(f"All flip: {x_test} -> {result_one}")
    assert np.array_equal(result_one, -x_test), "With bits=1, should negate all"
    print("✓ Test passed\n")

    # Test 5: Float32
    print("Test 5: NumPy Float32 Bit Flip")
    print("-" * 40)
    x_f32 = np.array([1.5, -2.3, 3.7, -0.5], dtype=np.float32)
    random_bits_f32 = np.array([1, 0, 1, 0], dtype=np.uint8)
    x_flipped_f32 = numpy_bit_flip(x_f32, random_bits_f32)

    expected_f32 = np.array([-1.5, -2.3, -3.7, -0.5], dtype=np.float32)

    print(f"Original:     {x_f32}")
    print(f"Random bits:  {random_bits_f32}")
    print(f"Flipped:      {x_flipped_f32}")
    print(f"Expected:     {expected_f32}")
    print(f"Match:        {np.allclose(x_flipped_f32, expected_f32)}")

    assert np.allclose(x_flipped_f32, expected_f32), "Float32 flip failed"
    print("✓ Test passed\n")

    # Test 6: cadna_round accepts list input
    print("Test 6: cadna_round List Input")
    print("-" * 40)
    gen_for_list = CADNARandomGenerator(seed=123)
    list_result = cadna_round([1.0, -2.0, 3.0], backend="numpy", random_gen=gen_for_list)
    print(f"Result: {list_result}")
    assert isinstance(list_result, np.ndarray), "cadna_round should return a NumPy array for NumPy backend"
    print("✓ Test passed\n")

    # Test 7: PyTorch backend, if available
    print("Test 7: PyTorch Backend")
    print("-" * 40)
    try:
        import torch

        x_torch = torch.tensor([1.5, -2.3, 3.7, -0.5], dtype=torch.float64)
        random_bits_torch = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
        x_flipped_torch = torch_bit_flip(x_torch, random_bits_torch)

        expected_torch = torch.tensor([1.5, -2.3, -3.7, 0.5], dtype=torch.float64)

        print(f"Original: {x_torch}")
        print(f"Flipped:  {x_flipped_torch}")
        print(f"Expected: {expected_torch}")
        print(f"Match:    {torch.allclose(x_flipped_torch, expected_torch)}")

        assert torch.allclose(x_flipped_torch, expected_torch), "PyTorch flip failed"
        print("✓ Test passed\n")
    except ImportError:
        print("⊘ PyTorch not available, skipping\n")

    # Test 8: JAX backend, if available
    print("Test 8: JAX Backend")
    print("-" * 40)
    try:
        import jax.numpy as jnp

        x_jax = jnp.array([1.5, -2.3, 3.7, -0.5], dtype=jnp.float32)
        random_bits_jax = jnp.array([0, 0, 1, 1], dtype=jnp.uint8)
        x_flipped_jax = jax_bit_flip(x_jax, random_bits_jax)

        expected_jax = jnp.array([1.5, -2.3, -3.7, 0.5], dtype=jnp.float32)

        print(f"Original: {x_jax}")
        print(f"Flipped:  {x_flipped_jax}")
        print(f"Expected: {expected_jax}")
        print(f"Match:    {bool(jnp.allclose(x_flipped_jax, expected_jax))}")

        assert bool(jnp.allclose(x_flipped_jax, expected_jax)), "JAX flip failed"
        print("✓ Test passed\n")
    except ImportError:
        print("⊘ JAX not available, skipping\n")

    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)