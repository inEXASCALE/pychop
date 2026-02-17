import torch
import numpy as np
import time
from pychop import LightChop, backend
from qtorch.quant import float_quantize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

backend('torch', 1)

sizes = [2000, 4000, 6000, 8000, 10000]

q_configs = {
    'exp5_man10': (5, 10),
    'exp8_man7': (8, 7)
}

times_pychop = {cfg: [] for cfg in q_configs}
times_qtorch = {cfg: [] for cfg in q_configs}

results_match = {cfg: [] for cfg in q_configs}

for size in sizes:
    print(f"\nMatrix size: {size}x{size}")

    A = torch.randn(size, size, device=device)

    for cfg_name, (exp, sig) in q_configs.items():
        print(f"  Configuration: {cfg_name}")

        # --- PyChop ---
        chopper = LightChop(exp_bits=exp, sig_bits=sig, subnormal=True, rmode=1)
        pychop_times = []
        C_chop = None
        for i in range(4):
            start = time.time()
            C_chop = chopper(A)
            if device=='cuda':
                torch.cuda.synchronize()
            pychop_times.append(time.time() - start)
        avg_pychop_time = np.mean(pychop_times[1:])
        times_pychop[cfg_name].append(avg_pychop_time)


        # --- QPytorch ---
        qtorch_times = []
        C_q_tensor = None
        for i in range(4):
            start = time.time()
            C_q_tensor = float_quantize(A, exp=exp, man=sig, rounding='nearest')
            if device=='cuda':
                torch.cuda.synchronize()
            qtorch_times.append(time.time() - start)
        avg_qtorch_time = np.mean(qtorch_times[1:])
        times_qtorch[cfg_name].append(avg_qtorch_time)

        match = torch.allclose(C_chop, C_q_tensor, rtol=1e-3, atol=1e-3)
        results_match[cfg_name].append(match)
        print(f"    Consistent PyChop vs QPyTorch: {match}")
        print(f"    PyChop avg time: {avg_pychop_time:.4f}s, QPyTorch avg time: {avg_qtorch_time:.4f}s")

np.savez('results.npz',
         sizes=sizes,
         q_configs=list(q_configs.keys()),
         times_pychop=times_pychop,
         times_qtorch=times_qtorch,
         results_match=results_match)
print("\nExperiment data saved to results.npz")
