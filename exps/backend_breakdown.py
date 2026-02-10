import os
import h5py
import pychop
from pychop import LightChop
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from jax import jit
from time import time
import gc

# 创建结果目录
os.makedirs("results", exist_ok=True)

# 参数设置
arr_sizes = [2000, 4000, 6000, 8000, 10000]
rounding_modes = ["Nearest (even)","Up","Down","Zero","Stochastic (prop)","Stochastic (uniform)"]
num_runs = 4  # 丢弃第一次 warm-up
operations = ["quantize_only", "quantize_matmul"]

backends = ["numpy", "torch_cpu"]
if torch.cuda.is_available():
    backends += ["torch_gpu"]
backends += ["jax_cpu", "jax_jit"]

# 存储平均时间和 throughput (G elements/s)
results_time = {op: {mode: pd.DataFrame(index=arr_sizes, columns=backends, dtype=float) 
                for mode in rounding_modes} for op in operations}
results_throughput = {op: {mode: pd.DataFrame(index=arr_sizes, columns=backends, dtype=float) 
                      for mode in rounding_modes} for op in operations}

for i, size in enumerate(arr_sizes):
    # 加载数据（假设为方阵）
    hd5 = h5py.File(f"data/random/A{i+1}.mat", 'r')
    A_np = np.asarray(hd5['A'][:]).astype(np.float64)  # 确保 double precision
    print(f"\nProcessing size {size}x{size}")

    for mode_name in range(len(rounding_modes)):
        print(f"  Rounding mode: {rounding_modes[mode_name]}")
        

        for op in operations:
            times = {be: [] for be in backends}

            for k in range(num_runs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                jax.clear_caches()

                # NumPy
                pychop.backend('numpy')
                ch_light = LightChop(exp_bits=5, sig_bits=10, rmode=mode_name+1)
                st = time()
                if op == "quantize_only":
                    _ = ch_light(A_np)
                else:
                    A_q = ch_light(A_np)
                    _ = np.matmul(A_q, A_q)
                times["numpy"].append(time() - st)

                # PyTorch CPU
                pychop.backend('torch')
                ch_light = LightChop(exp_bits=5, sig_bits=10, rmode=mode_name+1)
                A_th = torch.from_numpy(A_np).to('cpu')
                st = time()
                if op == "quantize_only":
                    _ = ch_light(A_th)
                else:
                    A_q = ch_light(A_th)
                    _ = torch.matmul(A_q, A_q)
                times["torch_cpu"].append(time() - st)

                # PyTorch GPU (if available)
                if "torch_gpu" in backends:
                    A_th_gpu = torch.from_numpy(A_np).to('cuda')
                    st = time()
                    if op == "quantize_only":
                        _ = ch_light(A_th_gpu)
                    else:
                        A_q = ch_light(A_th_gpu)
                        _ = torch.matmul(A_q, A_q)
                    torch.cuda.synchronize()  # 确保 GPU 完成
                    times["torch_gpu"].append(time() - st)

                # JAX CPU
                pychop.backend('jax')
                ch_light = LightChop(exp_bits=5, sig_bits=10, rmode=mode_name+1)
                A_jax = jnp.array(A_np)

                @jit
                def jitted_op(A):
                    if op == "quantize_only":
                        return ch_light(A)
                    else:
                        A_q = ch_light(A)
                        return jnp.matmul(A_q, A_q)

                # Warm-up JIT 编译（只在第一次）
                if k == 0:
                    _ = jitted_op(A_jax)
                    jax.block_until_ready(_)

                st = time()
                _ = jitted_op(A_jax)
                jax.block_until_ready(_)
                times["jax_cpu"].append(time() - st)  # 非 JIT
                times["jax_jit"].append(time() - st)  # JIT（从第二次起更快）

            # 计算平均时间（丢弃第一次）
            elements = size * size
            for be in backends:
                if len(times[be]) > 1:
                    avg_time = np.mean(times[be][1:])
                    throughput = elements / avg_time / 1e9  # G elements/s
                    results_time[op][mode_name].loc[size, be] = avg_time
                    results_throughput[op][mode_name].loc[size, be] = throughput

# 保存 CSV
for op in operations:
    for mode in rounding_modes:
        time_df = results_time[op][mode]
        throughput_df = results_throughput[op][mode]
        time_df.to_csv(f"results/{op}_{mode}_time.csv")
        throughput_df.to_csv(f"results/{op}_{mode}_throughput.csv")

print("\nBenchmark 完成！数据已保存到 results/ 目录。")