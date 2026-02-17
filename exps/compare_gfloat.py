import jax
import jax.numpy as jnp
import numpy as np
import time

from pychop import LightChop, backend
from gfloat import formats, RoundMode, round_ndarray

backend("jax", 1)

sizes = [2000, 4000, 6000, 8000, 10000]

gfloat_formats = {
    "binary16": formats.format_info_binary16,
    "bfloat16": formats.format_info_bfloat16,
}

times_pychop = {k: [] for k in gfloat_formats}
times_gfloat = {k: [] for k in gfloat_formats}
results_match = {k: [] for k in gfloat_formats}
rel_errors = {k: [] for k in gfloat_formats}

key = jax.random.PRNGKey(0)

for size in sizes:
    print(f"\nMatrix size: {size}x{size}")

    key, k1, k2 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (size, size))
    B = jax.random.normal(k2, (size, size))

    for name, fmt in gfloat_formats.items():
        print(f"  Format: {name}")

        sig_bits = fmt.precision - 1
        exp_bits = fmt.k - fmt.precision

        # -------------------
        # PyChop
        # -------------------
        ch = LightChop(
            exp_bits=exp_bits,
            sig_bits=sig_bits,
            rmode=1   # nearest-even
        )

        py_times = []
        for i in range(4):
            start = time.time()
            A_q = ch(A)
            B_q = ch(B)
            C_pychop = jnp.matmul(A_q, B_q)
            C_pychop.block_until_ready()
            py_times.append(time.time() - start)

        avg_py = np.mean(py_times[1:])
        times_pychop[name].append(avg_py)

        # -------------------
        # gfloat
        # -------------------
        gf_times = []
        for i in range(4):
            start = time.time()

            A_q_np = round_ndarray(fmt, np.asarray(A), rnd=RoundMode.TiesToEven)
            B_q_np = round_ndarray(fmt, np.asarray(B), rnd=RoundMode.TiesToEven)

            A_q = jnp.asarray(A_q_np)
            B_q = jnp.asarray(B_q_np)

            C_gfloat = jnp.matmul(A_q, B_q)
            C_gfloat.block_until_ready()

            gf_times.append(time.time() - start)

        avg_gf = np.mean(gf_times[1:])
        times_gfloat[name].append(avg_gf)

        diff = jnp.linalg.norm(C_pychop - C_gfloat)
        norm = jnp.linalg.norm(C_gfloat)
        rel_err = (diff / norm).item()

        match = bool(jnp.allclose(C_pychop, C_gfloat, rtol=1e-3, atol=1e-3))

        results_match[name].append(match)
        rel_errors[name].append(rel_err)

        print(f"    match: {match}")
        print(f"    rel_error: {rel_err:.4e}")
        print(f"    PyChop: {avg_py:.4f}s | gfloat: {avg_gf:.4f}s")

np.savez(
    "results.npz",
    sizes=sizes,
    formats=list(gfloat_formats.keys()),
    times_pychop=times_pychop,
    times_gfloat=times_gfloat,
    results_match=results_match,
    rel_errors=rel_errors,
)

