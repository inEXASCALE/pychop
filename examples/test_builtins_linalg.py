import numpy as np

# ---- Optional SciPy ----
try:
    import scipy.linalg as spla
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---- Optional JAX ----
try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

# ---- Optional Torch ----
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

import pychop
from pychop import Chop
from pychop.builtin import CPArray, CPJaxArray, CPTensor
from pychop.builtin.linalg import (
    eig, eigvals, eigh, eigvalsh,
    svd, qr, solve, inv, pinv,
    det, slogdet, matrix_rank, cond,
    norm, cholesky, trace, diagonal,
    expm, logm, sqrtm, polar,
    lu,
)


def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def main():
    # =========================
    # NumPy backend tests
    # =========================
    banner("NUMPY BACKEND")
    pychop.backend("numpy")
    half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

    A = CPArray(np.array([[1.0, 2.0], [3.0, 4.0]]), half)
    b = CPArray(np.array([1.0, 1.0]), half)
    S = CPArray(np.array([[2.0, 1.0], [1.0, 2.0]]), half)  # symmetric / SPD

    print("A:", A)
    print("S:", S)

    banner("numpy: eig/eigvals/eigh/eigvalsh")
    w, v = eig(A)
    print("eig types:", type(w), type(v))
    wv = eigvals(A)
    print("eigvals type:", type(wv))
    ws, vs = eigh(S)
    print("eigh types:", type(ws), type(vs))
    wsh = eigvalsh(S)
    print("eigvalsh type:", type(wsh))

    banner("numpy: svd/qr/solve/inv/pinv")
    U, Sig, Vt = svd(A)
    print("svd types:", type(U), type(Sig), type(Vt))
    Q, R = qr(A)
    print("qr types:", type(Q), type(R))
    x = solve(A, b)
    print("solve type:", type(x), "x:", x)
    Ai = inv(A)
    print("inv type:", type(Ai))
    Api = pinv(A)
    print("pinv type:", type(Api))

    banner("numpy: det/slogdet/rank/cond")
    d = det(A)
    sd = slogdet(A)
    rk = matrix_rank(A)
    cd = cond(A)
    print("det:", d, "type:", type(d))
    print("slogdet:", sd, "types:", type(sd[0]), type(sd[1]) if isinstance(sd, tuple) else type(sd))
    print("matrix_rank:", rk, "type:", type(rk))
    print("cond:", cd, "type:", type(cd))

    banner("numpy: norm/cholesky/trace/diagonal")
    nA = norm(A)
    L = cholesky(S)
    tr = trace(A)
    diag = diagonal(A)
    print("norm(A):", nA, "type:", type(nA))
    print("cholesky(S) type:", type(L))
    print("trace(A):", tr, "type:", type(tr))
    print("diagonal(A) type:", type(diag), "diag:", diag)

    banner("numpy: lu (SciPy required)")
    try:
        P, L, U = lu(A)  # requires SciPy
        print("lu types:", type(P), type(L), type(U))
    except Exception as e:
        print("lu skipped/failed:", repr(e))

    banner("numpy: expm/logm/sqrtm/polar (SciPy required)")
    if HAVE_SCIPY:
        E = expm(A)  # SciPy
        print("expm type:", type(E))
        try:
            LM = logm(A)
            print("logm type:", type(LM))
        except Exception as e:
            print("logm failed:", repr(e))
        try:
            SM = sqrtm(A)
            print("sqrtm type:", type(SM))
        except Exception as e:
            print("sqrtm failed:", repr(e))
        try:
            U_p, H_p = polar(A)
            print("polar types:", type(U_p), type(H_p))
        except Exception as e:
            print("polar failed:", repr(e))
    else:
        print("SciPy not installed; skipping expm/logm/sqrtm/polar tests.")

    # =========================
    # JAX backend tests
    # =========================
    banner("JAX BACKEND")
    if HAVE_JAX:
        pychop.backend("jax")
        half_j = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

        Aj = CPJaxArray(jnp.array([[1.0, 2.0], [3.0, 4.0]]), half_j)
        Sj = CPJaxArray(jnp.array([[2.0, 1.0], [1.0, 2.0]]), half_j)
        bj = CPJaxArray(jnp.array([1.0, 1.0]), half_j)

        banner("jax: logm/sqrtm/polar host fallback (requires SciPy)")
        if HAVE_SCIPY:
            for fn, name in [(logm, "logm"), (sqrtm, "sqrtm")]:
                try:
                    out = fn(Aj, allow_host_fallback=True)
                    print(f"jax {name}(host) type:", type(out))
                except Exception as e:
                    print(f"jax {name}(host) failed:", repr(e))
            try:
                U_h, H_h = polar(Aj, allow_host_fallback=True)
                print("jax polar(host) types:", type(U_h), type(H_h))
            except Exception as e:
                print("jax polar(host) failed:", repr(e))
        else:
            print("SciPy not installed; skipping JAX host fallback tests.")

        banner("jax: eig/eigh/svd/qr/solve/inv/norm/cholesky/trace/diagonal")
        wj, vj = eig(Aj)
        print("eig types:", type(wj), type(vj))
        wshj, vshj = eigh(Sj)
        print("eigh types:", type(wshj), type(vshj))
        Uj, Sjv, Vtj = svd(Aj)
        print("svd types:", type(Uj), type(Sjv), type(Vtj))
        Qj, Rj = qr(Aj)
        print("qr types:", type(Qj), type(Rj))
        xj = solve(Aj, bj)
        print("solve type:", type(xj))
        Aij = inv(Aj)
        print("inv type:", type(Aij))
        nj = norm(Aj)
        print("norm type:", type(nj), "value:", nj)
        Lj = cholesky(Sj)
        print("cholesky type:", type(Lj))
        trj = trace(Aj)
        print("trace type:", type(trj), "value:", trj)
        diagj = diagonal(Aj)
        print("diagonal type:", type(diagj))

        banner("jax: lu (jax.scipy required)")
        try:
            Pj, Lj2, Uj2 = lu(Aj)
            print("lu types:", type(Pj), type(Lj2), type(Uj2))
        except Exception as e:
            print("lu skipped/failed:", repr(e))

        banner("jax: expm (jax.scipy required)")
        try:
            Ej = expm(Aj)
            print("expm type:", type(Ej))
        except Exception as e:
            print("expm skipped/failed:", repr(e))

        banner("jax: logm/sqrtm/polar default (expected to fail)")
        for fn, name in [(logm, "logm"), (sqrtm, "sqrtm"), (polar, "polar")]:
            try:
                out = fn(Aj)
                print(name, "unexpectedly worked, type:", type(out))
            except Exception as e:
                print(name, "expected failure:", repr(e))
    else:
        print("JAX not installed; skipping JAX tests.")

    # =========================
    # Torch backend tests
    # =========================
    banner("TORCH BACKEND")
    if HAVE_TORCH:
        pychop.backend("torch")
        half_t = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

        At = CPTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), half_t)
        St = CPTensor(torch.tensor([[2.0, 1.0], [1.0, 2.0]]), half_t)
        bt = CPTensor(torch.tensor([1.0, 1.0]), half_t)

        banner("torch: logm/sqrtm/polar host fallback (requires SciPy)")
        if HAVE_SCIPY:
            for fn, name in [(logm, "logm"), (sqrtm, "sqrtm")]:
                try:
                    out = fn(At, allow_host_fallback=True)
                    print(f"torch {name}(host) type:", type(out))
                except Exception as e:
                    print(f"torch {name}(host) failed:", repr(e))
            try:
                U_h, H_h = polar(At, allow_host_fallback=True)
                print("torch polar(host) types:", type(U_h), type(H_h))
            except Exception as e:
                print("torch polar(host) failed:", repr(e))
        else:
            print("SciPy not installed; skipping Torch host fallback tests.")

        banner("torch: eig/eigh/svd/qr/solve/inv/pinv/norm/cholesky/trace/diagonal")
        ew, ev = eig(At)
        print("eig types:", type(ew), type(ev), "dtypes:", getattr(ew, "dtype", None), getattr(ev, "dtype", None))
        ewh, evh = eigh(St)
        print("eigh types:", type(ewh), type(evh))
        Ut, Sig_t, Vht = svd(At)
        print("svd types:", type(Ut), type(Sig_t), type(Vht))
        Qt, Rt = qr(At)
        print("qr types:", type(Qt), type(Rt))
        xt = solve(At, bt)
        print("solve type:", type(xt))
        Ait = inv(At)
        print("inv type:", type(Ait))
        Pit = pinv(At)
        print("pinv type:", type(Pit))
        nt = norm(At)
        print("norm type:", type(nt), "value:", nt)
        Lt = cholesky(St)
        print("cholesky type:", type(Lt))
        trt = trace(At)
        print("trace type:", type(trt), "value:", trt)
        diagt = diagonal(At)
        print("diagonal type:", type(diagt))

        banner("torch: lu (torch.linalg.lu_factor signature)")
        try:
            LU, piv = lu(At)
            print("lu_factor types:", type(LU), type(piv), "dtypes:", LU.dtype, piv.dtype)
        except Exception as e:
            print("lu_factor failed:", repr(e))

        banner("torch: expm (matrix_exp)")
        try:
            Et = expm(At)
            print("expm type:", type(Et))
        except Exception as e:
            print("expm failed:", repr(e))

        banner("torch: logm/sqrtm/polar host fallback (requires SciPy)")
        print("HAVE_SCIPY:", HAVE_SCIPY)
        if HAVE_SCIPY:
            out = sqrtm(At, allow_host_fallback=True)
            print("torch sqrtm(host) type:", type(out))
            out = logm(At, allow_host_fallback=True)
            print("torch logm(host) type:", type(out))
            U_h, H_h = polar(At, allow_host_fallback=True)
            print("torch polar(host) types:", type(U_h), type(H_h))
        else:
            print("SciPy not installed; skipping Torch host fallback tests.")

        banner("torch: logm/sqrtm/polar default (expected to fail)")
        for fn, name in [(logm, "logm"), (sqrtm, "sqrtm"), (polar, "polar")]:
            try:
                fn(At, allow_host_fallback=False)
                print(name, "unexpectedly worked")
            except Exception as e:
                print(name, "expected failure:", repr(e))

        banner("torch: logm/sqrtm/polar host fallback (requires SciPy)")
        for fn, name in [(logm, "logm"), (sqrtm, "sqrtm"), (polar, "polar")]:
            try:
                fn(At, allow_host_fallback=True)
            except Exception as e:
                print(name, "expected failure:", repr(e))

        banner("torch: lu_factor and lu_plu")
        from pychop.builtin.linalg import lu_factor, lu_plu

        try:
            LU2, piv2 = lu_factor(At)
            print("lu_factor types:", type(LU2), type(piv2))
        except Exception as e:
            print("lu_factor failed:", repr(e))

        try:
            P2, L2, U2 = lu_plu(At)
            print("lu_plu types:", type(P2), type(L2), type(U2))
        except Exception as e:
            print("lu_plu failed:", repr(e))
    else:
        print("Torch not installed; skipping Torch tests.")

    banner("ALL TESTS FINISHED")


if __name__ == "__main__":
    main()