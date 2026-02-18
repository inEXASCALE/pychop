"""
Benchmark script: Measure overhead of pychop emulated FP32 vs native FP32
in iterative refinement (Higham-style, LU once + repeated solve).

Tests matrix sizes: 2000 × 2000 and 5000 × 5000
Condition number: ~10^4
Backends: numpy, jax, torch (CPU)
Measurement: 4 runs → discard warm-up (first) → average of next 3
Saves results to 'results/pychop_overhead.csv'
"""

import numpy as np
import time
import pandas as pd
from scipy.linalg import qr, lu_factor, lu_solve

import pychop
from pychop import LightChop
import random
import torch

def fix_seed(seed: int = 42):
    """
    Fix random seed for reproducibility in PyTorch, NumPy, and Python.
    """
    random.seed(seed)                   # Python random
    np.random.seed(seed)                # NumPy
    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed(seed)        # PyTorch GPU
    torch.cuda.manual_seed_all(seed)    # All GPUs if multi-GPU

    # For deterministic behavior (may slow down some operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed(42)


def compare_solutions(x_native, x_emulated, tol: float = 1e-6) -> bool:
    """
    Compare native and emulated FP32 solutions.
    Returns True if they match within the given tolerance.
    
    Parameters
    ----------
    x_native : array-like
        Solution computed in native FP32
    x_emulated : array-like
        Solution computed in emulated FP32
    tol : float
        Relative tolerance for comparison

    Returns
    -------
    bool
        True if solutions match within tolerance, False otherwise
    """
    import numpy as np
    
    # Convert to float64 for stable comparison if needed
    x_native = np.asarray(x_native, dtype=np.float32)
    x_emulated = np.asarray(x_emulated, dtype=np.float32)

    # Maximum relative error
    max_rel_error = np.max(np.abs(x_native - x_emulated) / (np.abs(x_native) + 1e-12))

    return max_rel_error < tol


# ----------------------------------------------------------------------
#  Matrix / data generation
# ----------------------------------------------------------------------

def generate_ill_conditioned_matrix(n: int, cond: float = 1e4, seed: int = 42) -> np.ndarray:
    """Generate symmetric positive definite matrix with given 2-norm condition number."""
    np.random.seed(seed)
    A = np.random.randn(n, n)
    Q, _ = qr(A)
    s = np.logspace(0, -np.log10(cond), n)
    return Q @ np.diag(s) @ Q.T


def generate_system(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate A (ill-conditioned), b, and true solution x_true."""
    A = generate_ill_conditioned_matrix(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    return A, b, x_true


# ----------------------------------------------------------------------
#  Iterative Refinement Implementations
# ----------------------------------------------------------------------
def iterative_refinement_native(A: np.ndarray, b: np.ndarray,
                                 max_iter: int = 10, tol: float = 1e-6,
                                 backend: str = 'numpy') -> np.ndarray:
    """
    Native FP32 iterative refinement:
      - LU factorization once (in float32)
      - Repeated forward/back substitution using the same factors
    """
    if backend == 'numpy':
        A32 = A.astype(np.float32)
        b32 = b.astype(np.float32)
        lu, piv = lu_factor(A32)
        x = lu_solve((lu, piv), b32)
        lib_matmul = np.matmul
        lib_norm = np.linalg.norm

    elif backend == 'jax':
        try:
            import jax.numpy as jnp
            from jax.scipy.linalg import lu_factor as jax_lu_factor
            from jax.scipy.linalg import lu_solve as jax_lu_solve
        except ImportError:
            raise ImportError("JAX is required for backend='jax'")
        A32 = jnp.array(A, dtype=jnp.float32)
        b32 = jnp.array(b, dtype=jnp.float32)
        lu, piv = jax_lu_factor(A32)
        x = jax_lu_solve((lu, piv), b32)
        lib_matmul = jnp.matmul
        lib_norm = jnp.linalg.norm

    elif backend == 'torch':
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for backend='torch'")
        device = torch.device('cpu')
        A32 = torch.tensor(A, dtype=torch.float32, device=device)
        b32 = torch.tensor(b, dtype=torch.float32, device=device)
        
        LU, pivots = torch.linalg.lu_factor(A32)
        x = torch.linalg.lu_solve(LU, pivots, b32.unsqueeze(-1)).squeeze(-1)

        lib_matmul = torch.matmul
        lib_norm = torch.norm   # torch.norm is equivalent to linalg.norm for vectors

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    for _ in range(max_iter):
        r = b32 - lib_matmul(A32, x)
        if lib_norm(r) < tol:
            break

        if backend == 'numpy':
            d = lu_solve((lu, piv), r)
        elif backend == 'jax':
            d = jax_lu_solve((lu, piv), r)
        elif backend == 'torch':
            d = torch.linalg.lu_solve(LU, pivots, r.unsqueeze(-1)).squeeze(-1)

        x += d

    return x


def iterative_refinement_emulated(A: np.ndarray, b: np.ndarray,
                                  max_iter: int = 10, tol: float = 1e-6,
                                  backend: str = 'numpy') -> np.ndarray:
    """
    Emulated FP32 via pychop:
      - Data stored in double
      - Chop after every major operation (matmul, subtraction, addition, solve)
      - LU factorization once in double (typical Higham IR style)
    """
    pychop.backend(backend, 0)   # 0 or 1 depending on your pychop version/behavior; check doc
    ch = LightChop(exp_bits=8, sig_bits=23, rmode=1)  # IEEE fp32 emulation (nearest-even)

    if backend == 'numpy':
        lib = np
        A_in = ch(A.copy()).astype(np.float32)
        b_in = ch(b.copy()).astype(np.float32)
        lib_matmul = np.matmul
        lib_norm = np.linalg.norm

    elif backend == 'jax':
        try:
            import jax.numpy as jnp
            from jax.scipy.linalg import lu_factor as jax_lu_factor
            from jax.scipy.linalg import lu_solve as jax_lu_solve
        except ImportError:
            raise ImportError("JAX is required for backend='jax'")
        lib = jnp
        A_in = jnp.array(A)
        b_in = jnp.array(b)
        lib_matmul = jnp.matmul
        lib_norm = jnp.linalg.norm

    elif backend == 'torch':
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for backend='torch'")
        lib = torch
        device = torch.device('cpu')
        A_in = torch.tensor(A, dtype=torch.float32, device=device)
        b_in = torch.tensor(b, dtype=torch.float32, device=device)
        lib_matmul = torch.matmul
        lib_norm = torch.norm

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Initial solve in high precision, then chop
    if backend == 'numpy':
        lu, piv = lu_factor(A_in)
        x = ch(lu_solve((lu, piv), b_in))
    elif backend == 'jax':
        lu, piv = jax_lu_factor(A_in)
        x = ch(jax_lu_solve((lu, piv), b_in))
    elif backend == 'torch':
        LU, pivots = torch.linalg.lu_factor(A_in)
        x_init = torch.linalg.lu_solve(LU, pivots, b_in.unsqueeze(-1)).squeeze(-1)
        x = ch(x_init)

    for _ in range(max_iter):
        Ax = ch(lib_matmul(A_in, ch(x)))
        r = ch(b_in - Ax)

        if lib_norm(r) < tol:
            break

        if backend == 'numpy':
            d = ch(lu_solve((lu, piv), r))
        elif backend == 'jax':
            d = ch(jax_lu_solve((lu, piv), r))
        elif backend == 'torch':
            d_corr = torch.linalg.lu_solve(LU, pivots, r.unsqueeze(-1)).squeeze(-1)
            d = ch(d_corr)
        x = ch(x + d)

    return x


# ----------------------------------------------------------------------
#  Timing utility
# ----------------------------------------------------------------------

def measure_time(func, *args, num_runs: int = 4, **kwargs) -> float:
    """Run function num_runs times, discard first (warm-up), return average of the rest."""
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if i > 0:
            times.append(elapsed)
    return np.mean(times)


# ----------------------------------------------------------------------
#  Main experiment loop
# ----------------------------------------------------------------------

def run_benchmark():
    results = []
    sizes = [2000, 4000, 6000, 8000]

    for n in sizes:
        print(f"\nGenerating system for n = {n} ...")
        A, b, _ = generate_system(n)

        for backend in ['numpy', 'jax', 'torch']:
            print(f"  Backend: {backend}")
            # Compute solutions (not just timing)
            x_native = iterative_refinement_native(A, b, backend=backend)
            x_emul = iterative_refinement_emulated(A, b, backend=backend)

            # Compare correctness
            consistent = compare_solutions(x_native, x_emul)
            print(f"    Solutions match: {consistent}")

            try:
                t_native = measure_time(iterative_refinement_native, A, b, backend=backend)
                results.append({'size': n, 'backend': backend, 'mode': 'native', 'time': t_native})

                t_emul = measure_time(iterative_refinement_emulated, A, b, backend=backend)
                results.append({'size': n, 'backend': backend, 'mode': 'emulated', 'time': t_emul})

            except ImportError as e:
                print(f"    Skipped {backend}: {e}")
                continue
            except Exception as e:
                print(f"    Error in {backend}: {e}")
                continue

    df = pd.DataFrame(results)
    df.to_csv('results/pychop_overhead.csv', index=False)
    print("\nResults saved to 'pychop_overhead.csv'")
    print(df)


if __name__ == '__main__':
    run_benchmark()
