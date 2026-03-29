from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pychop

from .dispatch import ChopWrapSpec, chopwrap_call


# -------------------------
# Specs per backend
# -------------------------

def _numpy_specs():
    from .cparray import CPArray

    def is_cp(x): return type(x) is CPArray
    def unwrap(x): return np.asarray(x)

    def wrap(y, chopper):
        if isinstance(y, np.ndarray):
            return CPArray(y, chopper)
        return y

    return [ChopWrapSpec(unwrap=unwrap, wrap=wrap, is_cp=is_cp)]


def _jax_specs():
    from .cparray_jax import CPJaxArray

    try:
        import jax.numpy as jnp
    except Exception as e:
        raise RuntimeError("JAX not available but backend is 'jax'.") from e

    def is_cp(x): return type(x) is CPJaxArray
    def unwrap(x): return x.to_regular()

    def wrap(y, chopper):
        if hasattr(y, "shape") and hasattr(y, "dtype"):
            return CPJaxArray(jnp.asarray(y), chopper)
        return y

    return [ChopWrapSpec(unwrap=unwrap, wrap=wrap, is_cp=is_cp)]


def _torch_specs():
    from .cptensor import CPTensor

    try:
        import torch
    except Exception as e:
        raise RuntimeError("Torch not available but backend is 'torch'.") from e

    def is_cp(x): return type(x) is CPTensor
    def unwrap(x): return x.to_regular()

    def wrap(y, chopper):
        if isinstance(y, torch.Tensor):
            return CPTensor(y, chopper)
        return y

    return [ChopWrapSpec(unwrap=unwrap, wrap=wrap, is_cp=is_cp)]


def _specs_for_backend():
    b = pychop.get_backend()
    if b == "numpy":
        return _numpy_specs()
    if b == "jax":
        return _jax_specs()
    if b == "torch":
        return _torch_specs()
    raise ValueError(f"Unsupported backend: {b!r}")


# -------------------------
# Backend resolver
# -------------------------

def _resolve(func_numpy: Optional[Callable], func_jax: Optional[Callable], func_torch: Optional[Callable]) -> Callable:
    b = pychop.get_backend()
    if b == "numpy":
        if func_numpy is None:
            raise RuntimeError("This function is not available on NumPy backend (without SciPy).")
        return func_numpy
    if b == "jax":
        if func_jax is None:
            raise RuntimeError("This function is not available on JAX backend.")
        return func_jax
    if b == "torch":
        if func_torch is None:
            raise RuntimeError("This function is not available on Torch backend.")
        return func_torch
    raise ValueError(f"Unsupported backend: {b!r}")


def _call(func: Callable, *args, scalar_mode: str = "python", **kwargs):
    """
    Internal helper that applies unwrap->func->chop+wrap.

    Parameters
    ----------
    func : callable
        Backend function to call.
    *args, **kwargs
        Passed to the backend function.
    scalar_mode : {'python', 'cpfloat'}, default='python'
        Controls how numeric scalar outputs are returned:
        - 'python': return chopped python scalars (float/int)
        - 'cpfloat': return CPFloat instances for numeric scalars
    """
    specs = _specs_for_backend()
    return chopwrap_call(func, *args, specs=specs, scalar_mode=scalar_mode, **kwargs)


# =============================================================================
# Eigen / decompositions
# =============================================================================

def eig(A, *args, **kwargs):
    """Eigen-decomposition (general matrix)."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.eig,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.eig if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.eig if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def eigvals(A, *args, **kwargs):
    """Eigenvalues only (general matrix)."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.eigvals,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.eigvals if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.eigvals if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def eigh(A, *args, **kwargs):
    """Eigen-decomposition for Hermitian/symmetric matrices."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.eigh,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.eigh if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.eigh if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def eigvalsh(A, *args, **kwargs):
    """Eigenvalues only for Hermitian/symmetric matrices."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.eigvalsh,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.eigvalsh if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.eigvalsh if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def svd(A, *args, **kwargs):
    """Singular value decomposition."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.svd,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.svd if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.svd if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def qr(A, *args, **kwargs):
    """QR decomposition."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.qr,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.qr if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.qr if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def cholesky(A, *args, **kwargs):
    """Cholesky factorization."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.cholesky,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.cholesky if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.cholesky if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


# =============================================================================
# Solvers / inverses
# =============================================================================

def solve(A, B, *args, **kwargs):
    """Solve linear system A X = B."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.solve,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.solve if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.solve if b == "torch" else None),
    )
    return _call(func, A, B, *args, **kwargs)


def inv(A, *args, **kwargs):
    """Matrix inverse."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.inv,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.inv if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.inv if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


def pinv(A, *args, **kwargs):
    """Moore-Penrose pseudo-inverse."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.pinv,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.pinv if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.pinv if b == "torch" else None),
    )
    return _call(func, A, *args, **kwargs)


# =============================================================================
# Scalar-returning (enable scalar_mode="cpfloat")
# =============================================================================

def det(A, *args, **kwargs):
    """Determinant (returns CPFloat)."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.det,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.det if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.det if b == "torch" else None),
    )
    return _call(func, A, *args, scalar_mode="cpfloat", **kwargs)


def slogdet(A, *args, **kwargs):
    """Sign and log(abs(det(A))) (returns CPFloat scalars or tuple thereof)."""
    b = pychop.get_backend()
    func = _resolve(
        func_numpy=np.linalg.slogdet,
        func_jax=(__import__("jax.numpy", fromlist=["linalg"]).linalg.slogdet if b == "jax" else None),
        func_torch=(__import__("torch", fromlist=["linalg"]).linalg.slogdet if b == "torch" else None),
    )
    return _call(func, A, *args, scalar_mode="cpfloat", **kwargs)


def matrix_rank(A, *args, **kwargs):
    """Matrix rank (returns CPFloat)."""
    b = pychop.get_backend()

    func_numpy = np.linalg.matrix_rank
    func_jax = None
    func_torch = None

    if b == "jax":
        import jax.numpy as jnp
        func_jax = getattr(jnp.linalg, "matrix_rank", None)

    if b == "torch":
        import torch
        func_torch = getattr(torch.linalg, "matrix_rank", None)

    func = _resolve(func_numpy, func_jax, func_torch)
    return _call(func, A, *args, scalar_mode="cpfloat", **kwargs)


def cond(A, *args, **kwargs):
    """Condition number (returns CPFloat)."""
    b = pychop.get_backend()

    func_numpy = np.linalg.cond
    func_jax = None
    func_torch = None

    if b == "jax":
        import jax.numpy as jnp
        func_jax = getattr(jnp.linalg, "cond", None)

    if b == "torch":
        import torch
        func_torch = getattr(torch.linalg, "cond", None)

    func = _resolve(func_numpy, func_jax, func_torch)
    return _call(func, A, *args, scalar_mode="cpfloat", **kwargs)


def norm(x, *args, **kwargs):
    """Vector/matrix norm (returns CPFloat)."""
    b = pychop.get_backend()

    func_numpy = np.linalg.norm
    func_jax = None
    func_torch = None

    if b == "jax":
        import jax.numpy as jnp
        func_jax = jnp.linalg.norm

    if b == "torch":
        import torch
        func_torch = getattr(torch.linalg, "norm", None) or torch.norm

    func = _resolve(func_numpy, func_jax, func_torch)
    return _call(func, x, *args, scalar_mode="cpfloat", **kwargs)


def trace(A, *args, **kwargs):
    """Trace (returns CPFloat)."""
    b = pychop.get_backend()

    func_numpy = np.trace
    func_jax = None
    func_torch = None

    if b == "jax":
        import jax.numpy as jnp
        func_jax = jnp.trace

    if b == "torch":
        import torch
        func_torch = torch.trace

    func = _resolve(func_numpy, func_jax, func_torch)
    return _call(func, A, *args, scalar_mode="cpfloat", **kwargs)


# =============================================================================
# Non-scalar convenience
# =============================================================================

def diagonal(A, *args, **kwargs):
    """Diagonal extraction."""
    b = pychop.get_backend()

    func_numpy = np.diagonal
    func_jax = None
    func_torch = None

    if b == "jax":
        import jax.numpy as jnp
        func_jax = jnp.diagonal

    if b == "torch":
        import torch
        func_torch = torch.diagonal

    func = _resolve(func_numpy, func_jax, func_torch)
    return _call(func, A, *args, **kwargs)


# =============================================================================
# Advanced matrix functions (SciPy fallback for NumPy)
# =============================================================================

def expm(A, *args, prefer_scipy: bool = True, **kwargs):
    """Matrix exponential."""
    b = pychop.get_backend()
    specs = _specs_for_backend()

    if b == "numpy":
        if not prefer_scipy:
            raise RuntimeError("NumPy backend expm requires SciPy (prefer_scipy=True).")
        import scipy.linalg as spla
        return chopwrap_call(spla.expm, A, *args, specs=specs, **kwargs)

    if b == "jax":
        import jax.scipy.linalg as jsla
        if not hasattr(jsla, "expm"):
            raise RuntimeError("jax.scipy.linalg.expm not available in your JAX version.")
        return chopwrap_call(jsla.expm, A, *args, specs=specs, **kwargs)

    if b == "torch":
        import torch
        func = getattr(torch.linalg, "matrix_exp", None) or getattr(torch, "matrix_exp", None)
        if func is None:
            raise RuntimeError("torch.matrix_exp / torch.linalg.matrix_exp not available.")
        return chopwrap_call(func, A, *args, specs=specs, **kwargs)

    raise ValueError(f"Unsupported backend: {b!r}")

def _host_fallback_scipy_matrix_function(func_name: str, A, *, allow_host_fallback: bool):
    """
    Run scipy.linalg.<func_name> on host (CPU), then chop+wrap back to the active backend.

    This helper is used for optional fallbacks on JAX/Torch backends.

    Parameters
    ----------
    func_name : str
        SciPy function name under ``scipy.linalg``.
    A : CPArray or CPJaxArray or CPTensor
        Input matrix.
    allow_host_fallback : bool
        If False, raises an informative error.

    Returns
    -------
    CPArray or CPJaxArray or CPTensor
        Chopped+wrapped result in the current backend container.
    """
    if not allow_host_fallback:
        raise RuntimeError(
            f"{func_name} is not available on this backend. "
            f"Set allow_host_fallback=True to run scipy.linalg.{func_name} on host (CPU)."
        )

    try:
        import scipy.linalg as spla
    except Exception as e:
        raise RuntimeError(f"SciPy not available for host fallback: scipy.linalg.{func_name}.") from e

    if not hasattr(spla, func_name):
        raise RuntimeError(f"scipy.linalg.{func_name} not available in your SciPy version.")

    b = pychop.get_backend()

    # Unwrap A -> numpy array on host
    if b == "numpy":
        A_np = np.asarray(A)
    elif b == "torch":
        A_t = A.to_regular() if hasattr(A, "to_regular") else A
        A_np = A_t.detach().cpu().numpy()
    elif b == "jax":
        A_j = A.to_regular() if hasattr(A, "to_regular") else A
        A_np = np.asarray(A_j)
    else:
        raise ValueError(f"Unsupported backend: {b!r}")

    out_np = getattr(spla, func_name)(A_np)

    # Wrap back using known chopper from A (do NOT call chopwrap_call)
    specs = _specs_for_backend()
    spec = specs[0]
    return spec.wrap(out_np, A.chopper)


def logm(A, *args, prefer_scipy: bool = True, allow_host_fallback: bool = False, **kwargs):
    """
    Matrix logarithm.

    Backend behavior
    ----------------
    - NumPy backend:
        Uses ``scipy.linalg.logm`` (requires SciPy).
    - JAX/Torch backend:
        Optional SciPy-on-host fallback via ``allow_host_fallback=True``.

    Parameters
    ----------
    A : CPArray or CPJaxArray or CPTensor
        Input matrix.
    prefer_scipy : bool, default=True
        Required on NumPy backend.
    allow_host_fallback : bool, default=False
        If True on JAX/Torch, runs SciPy on CPU and wraps the result back.

    Returns
    -------
    CPArray or CPJaxArray or CPTensor
        Chopped+wrapped matrix logarithm.
    """
    b = pychop.get_backend()

    if b == "numpy":
        if not prefer_scipy:
            raise RuntimeError("NumPy backend logm requires SciPy (prefer_scipy=True).")
        import scipy.linalg as spla
        specs = _specs_for_backend()
        return chopwrap_call(spla.logm, A, *args, specs=specs, **kwargs)

    return _host_fallback_scipy_matrix_function("logm", A, allow_host_fallback=allow_host_fallback)


def sqrtm(A, *args, prefer_scipy: bool = True, allow_host_fallback: bool = False, **kwargs):
    """
    Matrix square root.

    Backend behavior
    ----------------
    - NumPy backend:
        Uses ``scipy.linalg.sqrtm`` (requires SciPy).
    - JAX/Torch backend:
        Optional SciPy-on-host fallback via ``allow_host_fallback=True``.

    Returns
    -------
    CPArray or CPJaxArray or CPTensor
        Chopped+wrapped matrix square root.
    """
    b = pychop.get_backend()

    if b == "numpy":
        if not prefer_scipy:
            raise RuntimeError("NumPy backend sqrtm requires SciPy (prefer_scipy=True).")
        import scipy.linalg as spla
        specs = _specs_for_backend()
        return chopwrap_call(spla.sqrtm, A, *args, specs=specs, **kwargs)

    return _host_fallback_scipy_matrix_function("sqrtm", A, allow_host_fallback=allow_host_fallback)


def polar(A, *args, prefer_scipy: bool = True, allow_host_fallback: bool = False, **kwargs):
    """
    Polar decomposition.

    Backend behavior
    ----------------
    - NumPy backend:
        Uses ``scipy.linalg.polar`` (requires SciPy).
    - JAX/Torch backend:
        Optional SciPy-on-host fallback via ``allow_host_fallback=True``.

    Returns
    -------
    (U, H) : tuple of (CPArray or CPJaxArray or CPTensor)
        Unitary/orthogonal factor U and Hermitian PSD factor H, chopped+wrapped.
    """
    b = pychop.get_backend()
    specs = _specs_for_backend()
    spec = specs[0]
    chopper = A.chopper

    if b == "numpy":
        if not prefer_scipy:
            raise RuntimeError("NumPy backend polar requires SciPy (prefer_scipy=True).")
        import scipy.linalg as spla
        if not hasattr(spla, "polar"):
            raise RuntimeError("scipy.linalg.polar not available (SciPy version too old).")
        return chopwrap_call(spla.polar, A, *args, specs=specs, **kwargs)

    if not allow_host_fallback:
        raise RuntimeError(
            "polar is only provided via SciPy in this wrapper. "
            "Set allow_host_fallback=True to run scipy.linalg.polar on host (CPU)."
        )

    try:
        import scipy.linalg as spla
    except Exception as e:
        raise RuntimeError("SciPy not available for host fallback: scipy.linalg.polar.") from e

    # Unwrap A -> numpy (host)
    if b == "torch":
        A_t = A.to_regular() if hasattr(A, "to_regular") else A
        A_np = A_t.detach().cpu().numpy()
    elif b == "jax":
        A_j = A.to_regular() if hasattr(A, "to_regular") else A
        A_np = np.asarray(A_j)
    else:
        raise ValueError(f"Unsupported backend: {b!r}")

    U_np, H_np = spla.polar(A_np, *args, **kwargs)
    return spec.wrap(U_np, chopper), spec.wrap(H_np, chopper)

# =============================================================================
# LU (backend-specific)
# =============================================================================

def lu(A, *args, prefer_scipy: bool = True, **kwargs):
    """
    LU decomposition.

    Notes
    -----
    - NumPy backend: SciPy required, returns (P, L, U).
    - JAX backend: jax.scipy.linalg.lu, returns (P, L, U).
    - Torch backend: torch.linalg.lu_factor, returns (LU, pivots).
    """
    b = pychop.get_backend()
    specs = _specs_for_backend()

    if b == "numpy":
        if not prefer_scipy:
            raise RuntimeError("NumPy backend LU requires SciPy (prefer_scipy=True).")
        import scipy.linalg as spla
        return chopwrap_call(spla.lu, A, *args, specs=specs, **kwargs)

    if b == "jax":
        import jax.scipy.linalg as jsla
        if not hasattr(jsla, "lu"):
            raise RuntimeError("jax.scipy.linalg.lu not available in your JAX version.")
        return chopwrap_call(jsla.lu, A, *args, specs=specs, **kwargs)

    if b == "torch":
        import torch
        func = getattr(torch.linalg, "lu_factor", None)
        if func is None:
            raise RuntimeError("torch.linalg.lu_factor not available in your torch version.")
        return chopwrap_call(func, A, *args, specs=specs, **kwargs)

    raise ValueError(f"Unsupported backend: {b!r}")




def lu_factor(A, *args, **kwargs):
    """
    Compute the LU factorization (backend-native signature).

    This function returns the backend's native LU-factor output:

    - NumPy backend (requires SciPy): ``scipy.linalg.lu_factor(A) -> (lu, piv)``
    - JAX backend: not provided (no stable equivalent in jax.scipy as a drop-in)
    - Torch backend: ``torch.linalg.lu_factor(A) -> (LU, pivots)``

    Parameters
    ----------
    A : CPArray or CPJaxArray or CPTensor
        Input matrix.
    *args, **kwargs
        Forwarded to the underlying backend function.

    Returns
    -------
    tuple
        LU factors and pivot indices, chopped+wrapped where applicable.

    Notes
    -----
    - For a SciPy-like ``(P, L, U)`` return, use :func:`lu_plu`.
    - For Torch, pivot indices are integer tensors; they will be wrapped by
      your CPTensor wrapper if you wrap all tensors uniformly. If you prefer
      pivots to remain plain integer tensors, that requires special-casing in
      the wrap spec.
    """
    b = pychop.get_backend()
    specs = _specs_for_backend()

    if b == "numpy":
        try:
            import scipy.linalg as spla
        except Exception as e:
            raise RuntimeError("SciPy not available for lu_factor on numpy backend.") from e
        if not hasattr(spla, "lu_factor"):
            raise RuntimeError("scipy.linalg.lu_factor not available.")
        return chopwrap_call(spla.lu_factor, A, *args, specs=specs, **kwargs)

    if b == "torch":
        import torch
        if not hasattr(torch.linalg, "lu_factor"):
            raise RuntimeError("torch.linalg.lu_factor not available in your torch version.")
        return chopwrap_call(torch.linalg.lu_factor, A, *args, specs=specs, **kwargs)

    raise RuntimeError("lu_factor is not provided on the JAX backend in this wrapper.")

def lu_plu(A, *args, **kwargs):
    """
    LU decomposition returning a SciPy-like (P, L, U) triple.

    Backend behavior
    ----------------
    - NumPy backend:
        Uses scipy.linalg.lu and returns (P, L, U).
    - JAX backend:
        Uses jax.scipy.linalg.lu and returns (P, L, U).
    - Torch backend:
        Tries (in order):
        1) torch.linalg.lu (if available, returns (P, L, U) in some versions)
        2) torch.lu (legacy; returns LU, pivots)
        3) torch.linalg.lu_factor (returns LU, pivots)
        Then unpacks (LU, pivots) manually to (P, L, U).

    Notes
    -----
    The manual unpack assumes 1-based pivot indices (common in torch).
    If your torch returns 0-based pivots, adjust `pi = piv - 1` to `pi = piv`.
    """
    b = pychop.get_backend()
    specs = _specs_for_backend()

    if b == "numpy":
        import scipy.linalg as spla
        return chopwrap_call(spla.lu, A, *args, specs=specs, **kwargs)

    if b == "jax":
        import jax.scipy.linalg as jsla
        if not hasattr(jsla, "lu"):
            raise RuntimeError("jax.scipy.linalg.lu not available in your JAX version.")
        return chopwrap_call(jsla.lu, A, *args, specs=specs, **kwargs)

    if b != "torch":
        raise ValueError(f"Unsupported backend: {b!r}")

    import torch

    # If torch.linalg.lu exists and returns (P,L,U), use it.
    linalg_lu = getattr(torch.linalg, "lu", None)
    if linalg_lu is not None:
        A_t = A.to_regular() if hasattr(A, "to_regular") else A
        out = linalg_lu(A_t, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 3:
            P_t, L_t, U_t = out
            chopper = A.chopper
            spec = specs[0]
            return spec.wrap(P_t, chopper), spec.wrap(L_t, chopper), spec.wrap(U_t, chopper)

    # Else compute LU, pivots via legacy torch.lu or torch.linalg.lu_factor
    legacy_lu = getattr(torch, "lu", None)
    if legacy_lu is not None:
        A_t = A.to_regular() if hasattr(A, "to_regular") else A
        LU_t, piv_t = legacy_lu(A_t, *args, **kwargs)
    else:
        if not hasattr(torch.linalg, "lu_factor"):
            raise RuntimeError("torch.linalg.lu_factor not available in your torch version.")
        A_t = A.to_regular() if hasattr(A, "to_regular") else A
        LU_t, piv_t = torch.linalg.lu_factor(A_t)

    # Manual unpack
    LU_t = LU_t
    piv_t = piv_t

    m, n = LU_t.shape[-2], LU_t.shape[-1]
    k = min(m, n)

    # L and U from packed LU
    eye_mk = torch.eye(m, k, dtype=LU_t.dtype, device=LU_t.device)
    L_t = torch.tril(LU_t[..., :m, :k], diagonal=-1) + eye_mk
    U_t = torch.triu(LU_t[..., :k, :n])

    # Build permutation matrix P from pivots
    batch_shape = piv_t.shape[:-1]
    I = torch.eye(m, dtype=LU_t.dtype, device=LU_t.device)
    P_t = I.expand(*batch_shape, m, m).clone()

    piv_int = piv_t.to(dtype=torch.int64)
    for i in range(k):
        # Common torch behavior: 1-based pivot indices
        pi = piv_int[..., i] - 1
        idx = torch.arange(m, device=LU_t.device)
        idx = idx.expand(*batch_shape, m).clone()

        idx_i = idx[..., i].clone()
        idx[..., i] = idx.gather(-1, pi.unsqueeze(-1)).squeeze(-1)
        idx.scatter_(-1, pi.unsqueeze(-1), idx_i.unsqueeze(-1))

        P_t = P_t.gather(-2, idx.unsqueeze(-1).expand(*batch_shape, m, m))

    # Wrap with known chopper (DO NOT call chopwrap_call here; it needs CP input)
    chopper = A.chopper
    spec = specs[0]
    return spec.wrap(P_t, chopper), spec.wrap(L_t, chopper), spec.wrap(U_t, chopper)