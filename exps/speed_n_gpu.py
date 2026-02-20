# This script is to compare performance of torch backend and numpy backend
import h5py
import pychop
from pychop import Chop
from pychop import FaultChop
import numpy as np
from time import time
import pandas as pd
import torch
import gc

rounding_modes = 6
num_runs = 11

arr_sizes = [2000, 4000, 6000, 8000, 10000] #[2**8, 2**9, 2**10, 2**11, 2**12, 2**13]
columns = ["Nearest (even)","Up","Down","Zero","Stochastic (prop)","Stochastic (uniform)"]

sizes = len(arr_sizes)

# Initialize runtime storage (for all runs) for pychop
runtimes_all_th_gpu = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th_gpu = np.zeros((sizes, rounding_modes))

# Initialize runtime storage (for all runs) for pychop
runtimes_all_th2_gpu = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th2_gpu = np.zeros((sizes, rounding_modes))

for i in range(sizes):
    hd5 = h5py.File(f"data/random/A{i+1}.mat", 'r')
    # print(f.keys()) # <KeysViewHDF5 ['X', 'y']>  
    # print() 
    X_th = np.asarray(hd5['A'][:])
    X_th = torch.from_numpy(X_th) # torch array
    X_th = X_th.to('cuda')
    print(f"\nData {i+1}")

    for j in range(rounding_modes):

        print(f"Rounding {columns[j]}")
        for k in range(num_runs):
            pychop.backend('torch')
            
            ch1 = Chop(exp_bits=5, sig_bits=10, rmode=j+1)
            ch2 = FaultChop('h', rmode=j+1)
            
            # GPU
            try:
                st = time()
                ch1(X_th)
                et = time()

                runtimes_all_th_gpu[i, j, k] = et - st
            except:
                runtimes_all_th_gpu[i, j, k] = -1
                print("CUDA allocation fails")
            
            try:
                st = time()
                ch2(X_th)
                et = time()

                runtimes_all_th2_gpu[i, j, k] = et - st
            except:
                runtimes_all_th2_gpu[i, j, k] = -1
                print("CUDA allocation fails")
            
            
            gc.collect()

        runtimes_avg_th_gpu[i, j] = np.mean(runtimes_all_th_gpu[i, j, 1:])
        runtimes_avg_th2_gpu[i, j] = np.mean(runtimes_all_th2_gpu[i, j, 1:])


runtimes_avg_th_gpu = pd.DataFrame(runtimes_avg_th_gpu, columns=columns)
runtimes_avg_th2_gpu = pd.DataFrame(runtimes_avg_th2_gpu, columns=columns)

runtimes_avg_th_gpu.index = arr_sizes[:sizes]
runtimes_avg_th2_gpu.index = arr_sizes[:sizes]
runtimes_avg_th_gpu.to_csv("results/pychop_runtimes_avg_th_gpu.csv", index=True, header=True)
runtimes_avg_th2_gpu.to_csv("results/pychop_runtimes_avg_th2_gpu.csv", index=True, header=True)