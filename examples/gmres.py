from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pychop

from pychop.builtin import CPArray, CPJaxArray, CPTensor
from pychop.builtin.linalg import norm, lu_factor

CPMat = Union[CPArray, CPJaxArray, CPTensor]
CPVec = Union[CPArray, CPJaxArray, CPTensor]

try:
    import scipy.linalg as spla  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _get_backend() -> str:
    """
    Return current backend as a string.

    Compatible with pychop versions that expose either:
    - pychop.backend() getter
    - pychop.get_backend()
    """
    if hasattr(pychop, "get_backend"):
        b = pychop.get_backend()
        if b is None:
            return "auto"
        return b
    # Older/newer API: backend() is getter/setter
    b = pychop.backend()
    if b is None:
        return "auto"
    return b


def _set_backend(b: str) -> None:
    """
    Set backend.

    Requires pychop.backend(<name>) to be the setter API.
    """
    if not hasattr(pychop, "backend"):
        raise RuntimeError("This pychop version does not expose pychop.backend(<name>) setter.")
    pychop.backend(b)


def _infer_backend_from_obj(x: object) -> Optional[str]:
    if isinstance(x, CPArray):
        return "numpy"
    if isinstance(x, CPJaxArray):
        return "jax"
    if isinstance(x, CPTensor):
        return "torch"
    return None


def _ensure_concrete_backend(*objs: object) -> str:
    """
    Ensure backend is concrete ('numpy'|'jax'|'torch'), not 'auto'.

    If current backend is 'auto', infer from the first CP* object in `objs` and
    set it globally via pychop.backend(inferred).

    Returns
    -------
    str
        Concrete backend.
    """
    b = _get_backend()
    if b != "auto":
        return b

    inferred = None
    for o in objs:
        inferred = _infer_backend_from_obj(o)
        if inferred is not None:
            break

    if inferred is None:
        raise ValueError(
            "pychop backend is 'auto' and cannot be inferred from inputs. "
            "Call pychop.backend('numpy'|'jax'|'torch') explicitly."
        )

    _set_backend(inferred)
    return inferred


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _as_scalar_float(x) -> float:
    try:
        return float(x)
    except Exception:
        if hasattr(x, "item"):
            return float(x.item())
        raise


def _zeros_like_vec(x: CPVec, n: int) -> CPVec:
    ch = x.chopper
    b = _ensure_concrete_backend(x)

    if b == "numpy":
        return CPArray(np.zeros((n,), dtype=float), ch)

    if b == "torch":
        import torch
        x_t = x.to_regular() if hasattr(x, "to_regular") else x
        z = torch.zeros((n,), dtype=x_t.dtype, device=x_t.device)
        return CPTensor(z, ch)

    if b == "jax":
        import jax.numpy as jnp
        x_j = x.to_regular() if hasattr(x, "to_regular") else x
        z = jnp.zeros((n,), dtype=jnp.asarray(x_j).dtype)
        return CPJaxArray(z, ch)

    raise ValueError(f"Unsupported backend: {b!r}")


def _matvec(A: Optional[CPMat], x: CPVec, matvec: Optional[Callable[[CPVec], CPVec]]) -> CPVec:
    if matvec is not None:
        return matvec(x)
    if A is None:
        raise ValueError("A is None but matvec is also None.")
    return A @ x


def _backends_stack_cols(cols):
    if len(cols) == 0:
        raise ValueError("No columns to stack.")
    b = _ensure_concrete_backend(cols[0])
    ch = cols[0].chopper

    if b == "numpy":
        M = np.stack([np.asarray(c) for c in cols], axis=1)
        return CPArray(M, ch)

    if b == "torch":
        import torch
        tensors = [(c.to_regular() if hasattr(c, "to_regular") else c) for c in cols]
        M = torch.stack(tensors, dim=1)
        return CPTensor(M, ch)

    if b == "jax":
        import jax.numpy as jnp
        arrs = [c.to_regular() if hasattr(c, "to_regular") else c for c in cols]
        M = jnp.stack(arrs, axis=1)
        return CPJaxArray(M, ch)

    raise ValueError(f"Unsupported backend: {b!r}")


def _backend_small_vec_like(x: CPVec, arr_np_1d: np.ndarray):
    ch = x.chopper
    b = _ensure_concrete_backend(x)

    if b == "numpy":
        return CPArray(arr_np_1d.astype(float), ch)

    if b == "torch":
        import torch
        x_t = x.to_regular() if hasattr(x, "to_regular") else x
        t = torch.tensor(arr_np_1d, dtype=x_t.dtype, device=x_t.device)
        return CPTensor(t, ch)

    if b == "jax":
        import jax.numpy as jnp
        t = jnp.asarray(arr_np_1d)
        return CPJaxArray(t, ch)

    raise ValueError(f"Unsupported backend: {b!r}")


# -----------------------------------------------------------------------------
# LU preconditioner
# -----------------------------------------------------------------------------

@dataclass
class LUPreconditioner:
    LU: Optional[object]
    piv: Optional[object]
    backend: str

    @classmethod
    def from_matrix(cls, A: CPMat) -> "LUPreconditioner":
        # Ensure backend is concrete before calling pychop.builtin.linalg.lu_factor
        b = _ensure_concrete_backend(A)

        if b == "jax":
            return cls(LU=None, piv=None, backend="jax")

        LU, piv = lu_factor(A)
        return cls(LU=LU, piv=piv, backend=b)

    def apply(self, v: CPVec) -> CPVec:
        b = _ensure_concrete_backend(v)

        if self.backend == "jax" or self.LU is None:
            return v

        if b != self.backend:
            raise ValueError(f"Preconditioner backend {self.backend!r} does not match vector backend {b!r}.")

        if self.backend == "numpy":
            if not HAVE_SCIPY:
                raise RuntimeError("NumPy LU preconditioner requires SciPy (scipy.linalg.lu_solve).")

            v_np = np.asarray(v)
            LU_np = np.asarray(self.LU)
            piv_np = np.asarray(self.piv).astype(np.int32)
            out_np = spla.lu_solve((LU_np, piv_np), v_np)
            return CPArray(out_np, v.chopper)

        if self.backend == "torch":
            import torch

            LU_t = self.LU.to_regular() if hasattr(self.LU, "to_regular") else self.LU
            piv_t = self.piv.to_regular() if hasattr(self.piv, "to_regular") else self.piv
            v_t = v.to_regular() if hasattr(v, "to_regular") else v

            rhs = v_t.unsqueeze(-1)

            if hasattr(torch, "lu_solve"):
                x_t = torch.lu_solve(rhs, LU_t, piv_t).squeeze(-1)
            elif hasattr(torch.linalg, "lu_solve"):
                try:
                    x_t = torch.linalg.lu_solve(LU_t, piv_t, rhs).squeeze(-1)
                except TypeError:
                    x_t = torch.linalg.lu_solve(LU_t, piv_t, rhs, left=True).squeeze(-1)
            else:
                raise RuntimeError("Torch LU preconditioner requires torch.lu_solve or torch.linalg.lu_solve.")

            return CPTensor(x_t, v.chopper)

        raise ValueError(f"Unsupported backend for LUPreconditioner: {self.backend!r}")


# -----------------------------------------------------------------------------
# GMRES (right-preconditioned)
# -----------------------------------------------------------------------------

def gmres(
    A: Optional[CPMat],
    b: CPVec,
    x0: Optional[CPVec] = None,
    *,
    matvec: Optional[Callable[[CPVec], CPVec]] = None,
    restart: int = 30,
    maxiter: int = 100,
    tol: float = 1e-6,
    preconditioner: Optional[LUPreconditioner] = None,
    build_lu_preconditioner: bool = False,
) -> Tuple[CPVec, Dict[str, object]]:
    _ensure_concrete_backend(A if A is not None else b, b)

    if A is None and matvec is None:
        raise ValueError("Provide either A or matvec.")

    n = int(np.asarray(b).shape[0])
    x = _zeros_like_vec(b, n) if x0 is None else x0

    if build_lu_preconditioner and preconditioner is None:
        if A is None:
            raise ValueError("build_lu_preconditioner=True requires an explicit matrix A.")
        preconditioner = LUPreconditioner.from_matrix(A)

    def apply_Minv(v: CPVec) -> CPVec:
        return preconditioner.apply(v) if preconditioner is not None else v

    def Aop(v: CPVec) -> CPVec:
        return _matvec(A, v, matvec)

    def Ahat(v: CPVec) -> CPVec:
        return Aop(apply_Minv(v))

    bnorm = _as_scalar_float(norm(b))
    if bnorm == 0.0:
        return x, {
            "converged": True,
            "num_outer": 0,
            "num_inner": 0,
            "res_norms": [0.0],
            "final_res_norm": 0.0,
        }

    res_hist = []
    total_inner = 0

    for outer in range(maxiter):
        r = b - Aop(x)
        beta = norm(r)
        beta_f = _as_scalar_float(beta)
        rel = beta_f / bnorm
        res_hist.append(rel)

        if rel <= tol:
            return x, {
                "converged": True,
                "num_outer": outer,
                "num_inner": total_inner,
                "res_norms": res_hist,
                "final_res_norm": rel,
            }

        V = []
        H = np.zeros((restart + 1, restart), dtype=float)

        v0 = r / beta
        V.append(v0)

        m_eff = 0
        x_restart0 = x

        for j in range(restart):
            total_inner += 1
            m_eff = j + 1

            w = Ahat(V[j])

            # Modified Gram-Schmidt (inner products on host)
            for i in range(j + 1):
                hij = np.dot(np.asarray(V[i]).conj(), np.asarray(w)).item()
                H[i, j] = float(hij)
                w = w - H[i, j] * V[i]

            h_next = _as_scalar_float(norm(w))
            H[j + 1, j] = h_next

            if h_next == 0.0:
                break

            V.append(w / h_next)

            # Solve small LS on host
            Hj = H[: j + 2, : j + 1]
            g = np.zeros((j + 2,), dtype=float)
            g[0] = beta_f

            Q, R = np.linalg.qr(Hj, mode="reduced")
            y = np.linalg.solve(R, Q.T @ g)

            Vm = _backends_stack_cols(V[: j + 1])
            y_cp = _backend_small_vec_like(b, y.reshape(-1))
            y_vec = Vm @ y_cp

            x_candidate = x_restart0 + apply_Minv(y_vec)

            # Residual estimate
            res_vec = g - Hj @ y
            rel_inner = float(np.linalg.norm(res_vec) / bnorm)
            if rel_inner <= tol:
                x = x_candidate
                return x, {
                    "converged": True,
                    "num_outer": outer,
                    "num_inner": total_inner,
                    "res_norms": res_hist,
                    "final_res_norm": rel_inner,
                }

        # End restart: final update
        Hj = H[: m_eff + 1, :m_eff]
        g = np.zeros((m_eff + 1,), dtype=float)
        g[0] = beta_f

        Q, R = np.linalg.qr(Hj, mode="reduced")
        y = np.linalg.solve(R, Q.T @ g)

        Vm = _backends_stack_cols(V[:m_eff])
        y_cp = _backend_small_vec_like(b, y.reshape(-1))
        y_vec = Vm @ y_cp
        x = x_restart0 + apply_Minv(y_vec)

    # Not converged
    r = b - Aop(x)
    rel_final = _as_scalar_float(norm(r)) / bnorm
    res_hist.append(rel_final)

    return x, {
        "converged": False,
        "num_outer": maxiter,
        "num_inner": total_inner,
        "res_norms": res_hist,
        "final_res_norm": rel_final,
    }


if __name__ == "__main__":
    import numpy as np
    from pychop import Chop
    from pychop.builtin import CPArray
    
    half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
    A = CPArray(np.array([[4.0, 1.0], [1.0, 3.0]]), half)
    b = CPArray(np.array([1.0, 2.0]), half)
    
    pc = LUPreconditioner.from_matrix(A)
    x, info = gmres(A, b, restart=5, maxiter=20, tol=1e-10, preconditioner=pc)
    print(x, info)