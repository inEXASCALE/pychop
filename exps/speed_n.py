# This script is to compare performance of torch backend and numpy backend
import h5py
import pychop
from pychop import LightChop
from pychop import Chop
import numpy as np
from time import time
import pandas as pd
import torch
import gc

rounding_modes = 6
num_runs = 11

arr_sizes = [2**6, 2**8, 2**10, 2**12, 2**14]
columns = ["Nearest (even)","Up","Down","Zero","Stochastic (prop)","Stochastic (uniform)"]

sizes = len(arr_sizes)

# Initialize runtime storage (for all runs) for pychop
runtimes_all_np = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_np = np.zeros((sizes, rounding_modes))

# Initialize runtime storage (for all runs) for pychop
runtimes_all_th = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th = np.zeros((sizes, rounding_modes))

# Initialize runtime storage (for all runs) for pychop
# runtimes_all_th_gpu = np.zeros((sizes, rounding_modes, num_runs))
# # Initialize average runtime storage (after discarding first run) for pychop
# runtimes_avg_th_gpu = np.zeros((sizes, rounding_modes))

# Initialize runtime storage (for all runs) for pychop
runtimes_all_np2 = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_np2 = np.zeros((sizes, rounding_modes))

# Initialize runtime storage (for all runs) for pychop
runtimes_all_th2 = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th2 = np.zeros((sizes, rounding_modes))

# Initialize runtime storage (for all runs) for pychop
# runtimes_all_th2_gpu = np.zeros((sizes, rounding_modes, num_runs))
# Initialize average runtime storage (after discarding first run) for pychop
# runtimes_avg_th2_gpu = np.zeros((sizes, rounding_modes))

for i in range(sizes):
    hd5 = h5py.File(f"data/random/A{i+1}.mat", 'r')
    # print(f.keys()) # <KeysViewHDF5 ['X', 'y']>  
    # print() 
    X = np.asarray(hd5['A'][:])
    print(f"\nData {i+1}")

    for j in range(rounding_modes):

        print(f"Rounding {columns[j]}")

        for k in range(num_runs):
            pychop.backend('numpy')
            
            ch1 = LightChop(exp_bits=5, sig_bits=10, rmode=j+1)
            ch2 = Chop('h', rmode=j+1)
            
            st = time()
            ch1(X)
            et = time()

            runtimes_all_np[i, j, k] = et - st

            st = time()
            ch2(X)
            et = time()

            runtimes_all_np2[i, j, k] = et - st

            pychop.backend('torch')
            X_th = torch.from_numpy(X) # torch array

            ch1 = LightChop(exp_bits=5, sig_bits=10, rmode=j+1)
            ch2 = Chop('h', rmode=j+1)

            st = time()
            ch1(X_th)
            et = time()

            runtimes_all_th[i, j, k] = et - st

            st = time()
            ch2(X_th)
            et = time()

            runtimes_all_th2[i, j, k] = et - st

            # GPU
            # X_th = X_th.to('cuda')

            # try:
            #     st = time()
            #     ch1(X_th)
            #     et = time()

            #     runtimes_all_th_gpu[i, j, k] = et - st
            # except:
            #     runtimes_all_th_gpu[i, j, k] = -1
            #     print("CUDA allocation fails")
                
            
            # try:
            #     st = time()
            #     ch2(X_th)
            #     et = time()

            #     runtimes_all_th2_gpu[i, j, k] = et - st
            # except:
            #     runtimes_all_th2_gpu[i, j, k] = -1
            #     print("CUDA allocation fails")
                

            gc.collect()

        runtimes_avg_np[i, j] = np.mean(runtimes_all_np[i, j, 1:])
        runtimes_avg_th[i, j] = np.mean(runtimes_all_th[i, j, 1:])
        # runtimes_avg_th_gpu[i, j] = np.mean(runtimes_all_th_gpu[i, j, 1:])

        runtimes_avg_np2[i, j] = np.mean(runtimes_all_np2[i, j, 1:])
        runtimes_avg_th2[i, j] = np.mean(runtimes_all_th2[i, j, 1:])
        # runtimes_avg_th2_gpu[i, j] = np.mean(runtimes_all_th2_gpu[i, j, 1:])


runtimes_avg_np = pd.DataFrame(runtimes_avg_np, columns=columns)
runtimes_avg_np2 = pd.DataFrame(runtimes_avg_np2, columns=columns)

runtimes_avg_th = pd.DataFrame(runtimes_avg_th, columns=columns)
runtimes_avg_th2 = pd.DataFrame(runtimes_avg_th2, columns=columns)

# runtimes_avg_th_gpu = pd.DataFrame(runtimes_avg_th_gpu, columns=columns)
# runtimes_avg_th2_gpu = pd.DataFrame(runtimes_avg_th2_gpu, columns=columns)

runtimes_avg_np.index = arr_sizes[:sizes]
runtimes_avg_th.index = arr_sizes[:sizes]
# runtimes_avg_th_gpu.index = arr_sizes[:sizes]
runtimes_avg_np2.index = arr_sizes[:sizes]
runtimes_avg_th2.index = arr_sizes[:sizes]
# runtimes_avg_th2_gpu.index = arr_sizes[:sizes]

runtimes_avg_np.to_csv("pychop_runtimes_avg_np.csv", index=True, header=True)
runtimes_avg_np2.to_csv("pychop_runtimes_avg_np2.csv", index=True, header=True)

runtimes_avg_th.to_csv("pychop_runtimes_avg_th.csv", index=True, header=True)
runtimes_avg_th2.to_csv("pychop_runtimes_avg_th2.csv", index=True, header=True)

# runtimes_avg_th_gpu.to_csv("pychop_runtimes_avg_th_gpu.csv", index=True, header=True)
# runtimes_avg_th2_gpu.to_csv("pychop_runtimes_avg_th2_gpu.csv", index=True, header=True)