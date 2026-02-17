import os, gc, h5py, numpy as np, pandas as pd, torch, jax, jax.numpy as jnp
from pychop import LightChop
import pychop
from jax import jit
from time import time

os.makedirs("results", exist_ok=True)

arr_sizes = [2000, 4000, 6000, 8000, 10000] 
rounding_modes = ["Nearest (even)", "Up", "Down", "Zero", "Stochastic (prop)", "Stochastic (uniform)"]
operations = ["quantize_only"]
num_runs = 4
backends = ["numpy", "torch_cpu", "jax_eager", "jax_jit"]

results_time = {op: {mode: pd.DataFrame(index=arr_sizes, columns=backends, dtype=float)
                     for mode in rounding_modes} for op in operations}
results_throughput = {op: {mode: pd.DataFrame(index=arr_sizes, columns=backends, dtype=float)
                           for mode in rounding_modes} for op in operations}

for i, size in enumerate(arr_sizes):
    with h5py.File(f"data/random/A{i+1}.mat", "r") as hd5:
        A_np = np.asarray(hd5["A"][:], dtype=np.float64)
    elements = size * size
    print(f"\nProcessing size {size} x {size}")

    for mode_idx, mode_name in enumerate(rounding_modes):
        print(f"  Rounding mode: {mode_name}")

        for op in operations:
            times = {be: [] for be in backends}

            # PyTorch CPU
            A_th = torch.from_numpy(A_np).to("cpu")

            # JAX
            A_jax = jnp.array(A_np)

            pychop.backend("jax")
            def jax_eager_op(A):
                ch = LightChop(exp_bits=8, sig_bits=7, rmode=mode_idx + 1)
                return ch(A)

            @jit
            def jax_jit_op(A):
                ch = LightChop(exp_bits=8, sig_bits=7, rmode=mode_idx + 1)
                return ch(A)

            _ = jax_jit_op(A_jax)  # warm-up
            jax.block_until_ready(_)

            for k in range(num_runs):
                gc.collect()

                pychop.backend("numpy")
                ch_np = LightChop(exp_bits=8, sig_bits=7, rmode=mode_idx + 1)
                st = time(); _ = ch_np(A_np); times["numpy"].append(time()-st)

                pychop.backend("torch")
                ch_th = LightChop(exp_bits=8, sig_bits=7, rmode=mode_idx + 1)
                st = time(); _ = ch_th(A_th); times["torch_cpu"].append(time()-st)

                pychop.backend("jax")
                st = time(); _ = jax_eager_op(A_jax); jax.block_until_ready(_)
                times["jax_eager"].append(time()-st)

                st = time(); _ = jax_jit_op(A_jax); jax.block_until_ready(_)
                times["jax_jit"].append(time()-st)

            for be in backends:
                if len(times[be]) > 1:
                    avg_time = np.mean(times[be][1:])
                    throughput = elements / avg_time / 1e9
                    results_time[op][mode_name].loc[size, be] = avg_time
                    results_throughput[op][mode_name].loc[size, be] = throughput

for op in operations:
    for mode in rounding_modes:
        safe_mode = mode.replace(" ", "_").replace("(", "").replace(")", "")
        results_time[op][mode].to_csv(f"results/{op}_{safe_mode}_cpu_time.csv")
        results_throughput[op][mode].to_csv(f"results/{op}_{safe_mode}_cpu_throughput.csv")

print("\nCPU benchmark 完成！")
