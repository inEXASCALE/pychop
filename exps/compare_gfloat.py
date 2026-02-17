import numpy as np
import torch
import jax
import jax.numpy as jnp
import time

from pychop import LightChop, backend as pychop_backend
from gfloat import formats, RoundMode, round_ndarray
jax.config.update("jax_enable_x64", True)

# ---------- Settings ----------
sizes = [2000, 4000, 6000, 8000, 10000]

gfloat_formats = {
    "binary16": formats.format_info_binary16,
    "bfloat16": formats.format_info_bfloat16,
}

results = {}

np.random.seed(42)
key = jax.random.PRNGKey(42)

# ---------- Loop over experiments ----------
for size in sizes:
    print(f"\nMatrix size: {size}x{size}")
    results[size] = {}

    # Generate random matrices
    A_np = np.random.randn(size, size)
    B_np = np.random.randn(size, size)  # For potential future use
    A_jax = jnp.array(A_np)
    A_torch = torch.tensor(A_np)

    for name, fmt in gfloat_formats.items():
        print(f"\n Format: {name}")
        results[size][name] = {}

        sig_bits = fmt.precision - 1
        exp_bits = fmt.k - fmt.precision

        # ---------- PyChop ----------
        for backend_name, X in [("numpy", A_np), ("jax", A_jax), ("torch", A_torch)]:
            pychop_backend(backend_name, 0)
            ch = LightChop(exp_bits=exp_bits, sig_bits=sig_bits, rmode=1)

            times = []
            X_q = None
            for _ in range(4):  # Run four times
                start = time.time()
                X_q = ch(X)
                if backend_name == "jax":
                    X_q.block_until_ready()
                elif backend_name == "torch":
                    # PyChop torch returns tensor, ensure correct device
                    if X_q.device.type != "cpu":
                        torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = np.mean(times[1:])  # Discard the first run and average the rest
            results[size][name][f"pychop_{backend_name}"] = {
                "quantized": X_q,
                "time": avg_time
            }
            print(f"    PyChop {backend_name} runtime: {avg_time}")

        print("\n")
        # ---------- gfloat ----------
        for g_backend, gX in [("numpy", A_np), ("jax", A_jax), ("torch", A_torch)]:
            times = []
            X_q_final = None
            for _ in range(4):
                start = time.time()
                X_q = round_ndarray(fmt, gX, rnd=RoundMode.TiesToEven)
                if g_backend == "jax":
                    X_q.block_until_ready()
                elif g_backend == "torch":
                    if isinstance(X_q, torch.Tensor):
                        X_q = X_q.detach().clone().to(A_torch.device)
                    else:
                        X_q = torch.from_numpy(np.array(X_q)).to(A_torch.device)

                X_q_final = X_q
                times.append(time.time() - start)

            avg_time = np.mean(times[1:])  # Discard the first run and average the rest
            results[size][name][f"gfloat_{g_backend}"] = {"quantized": X_q_final, "time": avg_time}
            print(f"    gfloat {g_backend} runtime: {avg_time}")

        # ---------- PyChop numpy vs gfloat numpy relative error ----------
        X_pychop_np = np.array(results[size][name]["pychop_numpy"]["quantized"])
        X_gfloat_np = np.array(results[size][name]["gfloat_numpy"]["quantized"])
        rel_error = np.linalg.norm(X_pychop_np - X_gfloat_np) / np.linalg.norm(X_gfloat_np)
        results[size][name]["rel_error_numpy"] = rel_error
        print(f"    {name} PyChop numpy vs gfloat numpy relative error: {rel_error:.4e}")

# ---------- Save results ----------
np.savez("quantize_results.npz", results=results)
print("\nSaved to quantize_results.npz")