import os
import gc
import h5py
import pychop
from pychop import Chop
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from jax import jit
from time import time

os.makedirs("results", exist_ok=True)

arr_sizes = [2000, 4000, 6000, 8000, 10000]
rounding_modes = [
    "Nearest (even)", "Up", "Down", "Zero",
    "Stochastic (prop)", "Stochastic (uniform)"
]
operations = ["quantize_only"]
num_runs = 4  # 丢弃第一次

backends = ["numpy", "torch_cpu"]
if torch.cuda.is_available():
    backends.append("torch_gpu")
backends += ["jax_eager", "jax_jit"]

results_time = {
    op: {
        mode: pd.DataFrame(index=arr_sizes, columns=backends, dtype=float)
        for mode in rounding_modes
    }
    for op in operations
}

results_throughput = {
    op: {
        mode: pd.DataFrame(index=arr_sizes, columns=backends, dtype=float)
        for mode in rounding_modes
    }
    for op in operations
}

# =========================================================
# Benchmark 
# =========================================================
for i, size in enumerate(arr_sizes):
    with h5py.File(f"data/random/A{i+1}.mat", "r") as hd5:
        A_np = np.asarray(hd5["A"][:], dtype=np.float64)

    elements = size * size
    print(f"\nProcessing size {size} x {size}")

    for mode_idx, mode_name in enumerate(rounding_modes):
        print(f"  Rounding mode: {mode_name}")

        for op in operations:
            times = {be: [] for be in backends}

            # -------------------------
            # PyTorch tensors 预创建
            # -------------------------
            A_th = torch.from_numpy(A_np).to("cpu")

            # -------------------------
            # JAX array 预创建
            # -------------------------
            A_jax = jnp.array(A_np)

            # =================================================
            # JAX：无状态 eager / JIT
            # =================================================
            pychop.backend("jax")

            def jax_eager_op(A):
                ch = Chop(
                    exp_bits=5,
                    sig_bits=10,
                    rmode=mode_idx + 1
                )
                return ch(A)

            @jit
            def jax_jit_op(A):
                ch = Chop(
                    exp_bits=5,
                    sig_bits=10,
                    rmode=mode_idx + 1
                )
                return ch(A)

            # JIT warm-up（只一次）
            _ = jax_jit_op(A_jax)
            jax.block_until_ready(_)

            for k in range(num_runs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # =====================
                # NumPy
                # =====================
                pychop.backend("numpy")
                ch_np = Chop(
                    exp_bits=5,
                    sig_bits=10,
                    rmode=mode_idx + 1
                )

                st = time()
                _ = ch_np(A_np)
                times["numpy"].append(time() - st)

                # =====================
                # PyTorch CPU
                # =====================
                pychop.backend("torch")
                ch_th = Chop(
                    exp_bits=5,
                    sig_bits=10,
                    rmode=mode_idx + 1
                )
                
                st = time()
                _ = ch_th(A_th)
                times["torch_cpu"].append(time() - st)

                # =====================
                # PyTorch GPU
                # =====================
                if "torch_gpu" in backends:
                    if "torch_gpu" in backends:
                        A_th = A_th.to("cuda")

                    st = time()
                    _ = ch_th(A_th)
                    torch.cuda.synchronize()
                    times["torch_gpu"].append(time() - st)

                # =====================
                # JAX eager
                # =====================
                pychop.backend("jax")
                st = time()
                _ = jax_eager_op(A_jax)
                jax.block_until_ready(_)
                times["jax_eager"].append(time() - st)

                # =====================
                # JAX JIT
                # =====================
                st = time()
                _ = jax_jit_op(A_jax)
                jax.block_until_ready(_)
                times["jax_jit"].append(time() - st)

            # =================================================
            # 统计（丢弃第一次）
            # =================================================
            for be in backends:
                if len(times[be]) > 1:
                    avg_time = np.mean(times[be][1:])
                    throughput = elements / avg_time / 1e9
                    results_time[op][mode_name].loc[size, be] = avg_time
                    results_throughput[op][mode_name].loc[size, be] = throughput

# =========================================================
# 保存 CSV
# =========================================================
for op in operations:
    for mode in rounding_modes:
        results_time[op][mode].to_csv(
            f"results/{op}_{mode}_time.csv"
        )
        results_throughput[op][mode].to_csv(
            f"results/{op}_{mode}_throughput.csv"
        )

print("\nBenchmark 完成！结果已保存到 results/ 目录。")
