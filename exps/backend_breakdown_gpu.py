# https://docs.jax.dev/en/latest/faq.html

import os, gc, h5py, torch
import pychop
from pychop import LightChop
from time import time
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)

arr_sizes = [2000, 4000, 6000, 8000, 10000]  
rounding_modes = ["Nearest (even)", "Up", "Down", "Zero", "Stochastic (prop)", "Stochastic (uniform)"]
operations = ["quantize_only"]
num_runs = 4
backends = ["torch_gpu"]

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

            # PyTorch GPU
            A_th_gpu = torch.from_numpy(A_np).to("cuda")
            pychop.backend("torch")
            ch_th = LightChop(exp_bits=8, sig_bits=7, rmode=mode_idx + 1)

            for k in range(num_runs):
                gc.collect()
                torch.cuda.empty_cache()

                st = time(); _ = ch_th(A_th_gpu); torch.cuda.synchronize()
                times["torch_gpu"].append(time()-st)

            avg_time = np.mean(times["torch_gpu"][1:])
            throughput = elements / avg_time / 1e9
            results_time[op][mode_name].loc[size, "torch_gpu"] = avg_time
            results_throughput[op][mode_name].loc[size, "torch_gpu"] = throughput

for op in operations:
    for mode in rounding_modes:
        safe_mode = mode.replace(" ", "_").replace("(", "").replace(")", "")
        results_time[op][mode].to_csv(f"results/{op}_{safe_mode}_gpu_time.csv")
        results_throughput[op][mode].to_csv(f"results/{op}_{safe_mode}_gpu_throughput.csv")

print("\nGPU benchmark 完成！")
