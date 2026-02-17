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

key = jax.random.PRNGKey(0)

for size in sizes:
    print(f"\nMatrix size: {size}x{size}")

    key, k1 = jax.random.split(key)
    A = jax.random.normal(k1, (size, size))

    for name, fmt in gfloat_formats.items():
        print(f"  Format: {name}")

        sig_bits = fmt.precision - 1
        exp_bits = fmt.k - fmt.precision

        ch = LightChop(
            exp_bits=exp_bits,
            sig_bits=sig_bits,
            rmode=1
        )

        py_times = []
        for i in range(5):
            start = time.time()
            A_q = ch(A)
            A_q.block_until_ready()
            py_times.append(time.time() - start)

        avg_py = np.mean(py_times[1:])
        times_pychop[name].append(avg_py)

        gf_times = []
        for i in range(5):
            start = time.time()
            A_q_np = round_ndarray(fmt, np.asarray(A), rnd=RoundMode.TiesToEven)
            A_q = jnp.asarray(A_q_np)
            A_q.block_until_ready()
            gf_times.append(time.time() - start)

        avg_gf = np.mean(gf_times[1:])
        times_gfloat[name].append(avg_gf)

        print(f"    PyChop quantize: {avg_py:.4f}s | gfloat quantize: {avg_gf:.4f}s")
        print(f"    ratio (pychop / gfloat): {avg_py / avg_gf:.2f}x")

np.savez(
    "quantize_only_results.npz",
    sizes=sizes,
    formats=list(gfloat_formats.keys()),
    times_pychop=times_pychop,
    times_gfloat=times_gfloat,
)