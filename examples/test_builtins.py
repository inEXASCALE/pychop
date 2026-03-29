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

# If you added chopwrap into pychop/builtin/cparray_jax.py as suggested:
try:
    from pychop.builtin.cparray_jax import chopwrap as jax_chopwrap
    HAVE_JAX_CHOPWRAP = True
except Exception:
    HAVE_JAX_CHOPWRAP = False


def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def assert_is_cparray(x, name):
    assert isinstance(x, CPArray), f"{name} should be CPArray, got {type(x)}"


def assert_is_cpjaxarray(x, name):
    assert isinstance(x, CPJaxArray), f"{name} should be CPJaxArray, got {type(x)}"


def assert_is_cptensor(x, name):
    assert isinstance(x, CPTensor), f"{name} should be CPTensor, got {type(x)}"


def main():
    # =========================
    # NumPy backend tests
    # =========================
    pychop.backend("numpy")
    half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

    banner("1) NumPy: CPArray basic ops (ufunc path)")
    p = CPArray(np.array([10.0, 20.0, 30.0]), half)
    q = CPArray(np.array([1.0, 2.0, 3.0]), half)
    print("p:", p)
    print("q:", q)

    r = p - q
    print("r = p - q:", r)
    assert_is_cparray(r, "r")

    plain = np.array([0.5, 1.5, 2.5])
    s = p * plain
    print("s = p * plain:", s)
    assert_is_cparray(s, "s")

    banner("2) NumPy: Matmul (@)")
    M = CPArray(np.random.rand(3, 4), half)
    N = CPArray(np.random.rand(4, 2), half)
    P = M @ N
    print("P.shape:", P.shape)
    assert_is_cparray(P, "P")

    banner("3) NumPy: np.linalg.eig / svd / qr / solve / inv")
    A = CPArray(np.array([[1.0, 2.0], [3.0, 4.0]]), half)

    w, v = np.linalg.eig(A)
    print("eigvals:", w, "type:", type(w))
    print("eigvecs:", v, "type:", type(v))
    assert_is_cparray(w, "eigvals w")
    assert_is_cparray(v, "eigvecs v")

    U, S, Vt = np.linalg.svd(A)
    print("svd U type:", type(U), "S type:", type(S), "Vt type:", type(Vt))
    assert_is_cparray(U, "svd U")
    assert_is_cparray(S, "svd S")
    assert_is_cparray(Vt, "svd Vt")

    Q, R = np.linalg.qr(A)
    print("qr Q type:", type(Q), "R type:", type(R))
    assert_is_cparray(Q, "qr Q")
    assert_is_cparray(R, "qr R")

    b = CPArray(np.array([1.0, 1.0]), half)
    x = np.linalg.solve(A, b)
    print("solve x:", x, "type:", type(x))
    assert_is_cparray(x, "solve x")

    Ai = np.linalg.inv(A)
    print("inv Ai type:", type(Ai))
    assert_is_cparray(Ai, "inv Ai")

    banner("4) SciPy: lu/qr/svd (direct) + wrapper")
    if HAVE_SCIPY:
        try:
            Pm, Lm, Um = spla.lu(A)
            print("direct spla.lu worked; types:", type(Pm), type(Lm), type(Um))
        except Exception as e:
            print("direct spla.lu FAILED:", repr(e))

        try:
            Qm, Rm = spla.qr(A)
            print("direct spla.qr worked; types:", type(Qm), type(Rm))
        except Exception as e:
            print("direct spla.qr FAILED:", repr(e))

        try:
            Um, Sm, Vhm = spla.svd(A)
            print("direct spla.svd worked; types:", type(Um), type(Sm), type(Vhm))
        except Exception as e:
            print("direct spla.svd FAILED:", repr(e))

        # Wrapper examples (guarantee chopped+wrapped outputs)
        def lu_wrapper(a: CPArray, *args, **kwargs):
            a0 = np.asarray(a)
            P0, L0, U0 = spla.lu(a0, *args, **kwargs)
            return CPArray(P0, a.chopper), CPArray(L0, a.chopper), CPArray(U0, a.chopper)

        P2, L2, U2 = lu_wrapper(A)
        print("wrapper lu types:", type(P2), type(L2), type(U2))
        assert_is_cparray(P2, "wrapper P2")
        assert_is_cparray(L2, "wrapper L2")
        assert_is_cparray(U2, "wrapper U2")
    else:
        print("SciPy not installed; skipping SciPy tests.")

    # =========================
    # JAX backend tests
    # =========================
    banner("5) JAX: CPJaxArray ops + jnp.linalg + (optional) chopwrap")
    if HAVE_JAX:
        pychop.backend("jax")
        half_jax = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

        x = CPJaxArray(jnp.array([10.0, 20.0, 30.0]), half_jax)
        y = CPJaxArray(jnp.array([1.0, 2.0, 3.0]), half_jax)
        print("x:", x)
        print("y:", y)

        z = x - y
        print("z = x - y:", z)
        assert_is_cpjaxarray(z, "jax z")

        A_j = CPJaxArray(jnp.array([[1.0, 2.0], [3.0, 4.0]]), half_jax)

        # Direct calls: should run if __jax_array__ exists, but outputs are usually jax.Array
        w0, v0 = jnp.linalg.eig(A_j)
        print("direct jnp.linalg.eig types:", type(w0), type(v0))

        U0, S0, Vt0 = jnp.linalg.svd(A_j)
        print("direct jnp.linalg.svd types:", type(U0), type(S0), type(Vt0))

        Q0, R0 = jnp.linalg.qr(A_j)
        print("direct jnp.linalg.qr types:", type(Q0), type(R0))

        # Chopped+wrapped outputs (optional)
        if HAVE_JAX_CHOPWRAP:
            wj, vj = jax_chopwrap(jnp.linalg.eig, A_j)
            print("chopwrap eig types:", type(wj), type(vj))
            assert_is_cpjaxarray(wj, "chopwrap eig wj")
            assert_is_cpjaxarray(vj, "chopwrap eig vj")

            Uj, Sj, Vtj = jax_chopwrap(jnp.linalg.svd, A_j)
            print("chopwrap svd types:", type(Uj), type(Sj), type(Vtj))
            assert_is_cpjaxarray(Uj, "chopwrap svd Uj")
            assert_is_cpjaxarray(Sj, "chopwrap svd Sj")
            assert_is_cpjaxarray(Vtj, "chopwrap svd Vtj")

            Qj, Rj = jax_chopwrap(jnp.linalg.qr, A_j)
            print("chopwrap qr types:", type(Qj), type(Rj))
            assert_is_cpjaxarray(Qj, "chopwrap qr Qj")
            assert_is_cpjaxarray(Rj, "chopwrap qr Rj")
        else:
            print("jax_chopwrap not found; skipping chopped+wrapped JAX output tests.")
    else:
        print("JAX not installed; skipping JAX tests.")

    # =========================
    # Torch backend tests
    # =========================
    banner("6) Torch: CPTensor ops + torch.linalg (svd/qr/lu/solve/inv)")
    if HAVE_TORCH:
        pychop.backend("torch")
        half_th = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

        a = CPTensor(torch.tensor([10.0, 20.0, 30.0]), half_th)
        b = CPTensor(torch.tensor([1.0, 2.0, 3.0]), half_th)
        print("a:", a)
        print("b:", b)

        c = a - b
        print("c = a - b:", c)
        assert_is_cptensor(c, "torch c")

        reg = torch.tensor([0.5, 1.5, 2.5])
        d = a * reg
        print("d = a * reg:", d)
        assert_is_cptensor(d, "torch d")

        A_th = CPTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), half_th)

        # QR
        try:
            Q, R = torch.linalg.qr(A_th)
            print("torch.linalg.qr types:", type(Q), type(R), "dtypes:", Q.dtype, R.dtype)
        except Exception as e:
            print("torch.linalg.qr(CPTensor) FAILED:", repr(e))

        # SVD
        try:
            U, S, Vh = torch.linalg.svd(A_th)
            print("torch.linalg.svd types:", type(U), type(S), type(Vh), "dtypes:", U.dtype, S.dtype, Vh.dtype)
        except Exception as e:
            print("torch.linalg.svd(CPTensor) FAILED:", repr(e))

        # LU (note: torch.linalg.lu_factor / lu solve APIs differ by version)
        try:
            # torch.linalg.lu_factor returns (LU, pivots)
            LU, piv = torch.linalg.lu_factor(A_th)
            print("torch.linalg.lu_factor types:", type(LU), type(piv), "dtypes:", LU.dtype, piv.dtype)
        except Exception as e:
            print("torch.linalg.lu_factor(CPTensor) FAILED:", repr(e))

        # Solve
        try:
            rhs = CPTensor(torch.tensor([1.0, 1.0]), half_th)
            x = torch.linalg.solve(A_th, rhs)
            print("torch.linalg.solve type:", type(x), "dtype:", x.dtype)
        except Exception as e:
            print("torch.linalg.solve(CPTensor) FAILED:", repr(e))

        # Inverse
        try:
            Ai = torch.linalg.inv(A_th)
            print("torch.linalg.inv type:", type(Ai), "dtype:", Ai.dtype)
        except Exception as e:
            print("torch.linalg.inv(CPTensor) FAILED:", repr(e))

        # Eig (complex output; may fail if Chop can't handle complex)
        try:
            ew, ev = torch.linalg.eig(A_th)
            print("torch.linalg.eig types:", type(ew), type(ev), "dtypes:", ew.dtype, ev.dtype)
        except Exception as e:
            print("torch.linalg.eig(CPTensor) FAILED:", repr(e))
            print("Note: eig is complex; your Chop backend may not support complex chopping.")
    else:
        print("Torch not installed; skipping Torch tests.")

    banner("ALL EXTENDED TESTS COMPLETED")


if __name__ == "__main__":
    main()