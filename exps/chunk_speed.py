from pychop import LightChop
import numpy as np

X_np = np.random.randn(10000, 5000) # Numpy array

chunk_sizes = np.arange(200, 5000, 500)

runtime_avg = list()
for size in chunk_sizes:
    ch = LightChop(exp_bits=5, sig_bits=10, chunk_size=size)
    runtime = list()
    
    for i in range(11):
        st = time()
        A_emu = ch(X_np)
        et = time()
        runtime.append(et - st)

    runtime_avg.append(np.mean(runtime[1:]))
    print(runtime_avg[-1])
    