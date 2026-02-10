# requirements: pip install numpy torch jax[cpu]   或 jax[cuda12_pip] 如果有 NVIDIA GPU
# 如果用 TPU/AMD 等，需要对应安装方式

# module load cuda/12.1
# module load cudnn/9.12.0_cuda12

import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import jit, random

# ----------------------- 配置部分 -----------------------
N = 4096                # 矩阵大小 N×N，越大越能看出差距（但别太大把显存爆了）
REPEATS = 20            # 热身/稳定测量重复次数
WARMUP = 5              # 前几次不算（compilation / cache）

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if "cuda" in device:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 随机种子，保证输入相同
key = random.PRNGKey(42)
np.random.seed(42)
torch.manual_seed(42)

# 生成数据（都在 CPU 上先生成，再搬到对应设备）
A_np = np.random.randn(N, N).astype(np.float32)
B_np = np.random.randn(N, N).astype(np.float32)

# NumPy（纯 CPU）
A_numpy = A_np.copy()
B_numpy = B_np.copy()

# PyTorch
if device == "cuda":
    A_torch = torch.from_numpy(A_np).to(device)
    B_torch = torch.from_numpy(B_np).to(device)
else:
    A_torch = torch.from_numpy(A_np)
    B_torch = torch.from_numpy(B_np)

# JAX（默认放默认设备，通常是 GPU 如果装了 cuda 版）
A_jax = jax.device_put(A_np)
B_jax = jax.device_put(B_np)

print(f"Matrix shape: {N} × {N}\n")

def benchmark_numpy():
    def op():
        return A_numpy @ B_numpy
    # warmup
    for _ in range(WARMUP):
        _ = op()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        C = op()
        if device == "cuda": torch.cuda.synchronize()  
        times.append(time.perf_counter() - t0)
    return min(times), np.median(times), np.mean(times)

def benchmark_torch():
    def op():
        return A_torch @ B_torch
    # warmup
    for _ in range(WARMUP):
        _ = op()
    if device == "cuda": torch.cuda.synchronize()
    print(f"use device:{device}")
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        C = op()
        if device == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times), np.median(times), np.mean(times)

@jit
def jax_matmul(a, b):
    return a @ b

def benchmark_jax():
    _ = jax_matmul(A_jax, B_jax).block_until_ready()

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        C = jax_matmul(A_jax, B_jax).block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times), np.median(times), np.mean(times)

print("NumPy (CPU):")
t_min, t_med, t_mean = benchmark_numpy()
print(f"  min   = {t_min*1000:6.2f} ms")
print(f"  median= {t_med*1000:6.2f} ms")
print(f"  mean  = {t_mean*1000:6.2f} ms\n")

print("PyTorch:")
t_min, t_med, t_mean = benchmark_torch()
print(f"  min   = {t_min*1000:6.2f} ms")
print(f"  median= {t_med*1000:6.2f} ms")
print(f"  mean  = {t_mean*1000:6.2f} ms\n")

print("JAX (jit-ed):")
t_min, t_med, t_mean = benchmark_jax()
print(f"  min   = {t_min*1000:6.2f} ms")
print(f"  median= {t_med*1000:6.2f} ms")
print(f"  mean  = {t_mean*1000:6.2f} ms")