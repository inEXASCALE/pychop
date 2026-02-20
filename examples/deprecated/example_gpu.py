
import sys
# appending a path
sys.path.append('../')


import numpy as np
import torch
import pychop
from time import time
from pychop.chop import Chop

from pychop.lightchop import LightChop
# from pychop.quant import quant
from time import time

if __name__ == '__main__':
    np.random.seed(0)

    X_np = np.random.randn(7000, 5000) # Numpy array
    X_th = torch.Tensor(X_np) # torch array

    pychop.backend('torch', 1) # print information

    ch = Chop('h')
    st = time()
    ch(X_th)
    print("runtime 1:", time() - st)


    ch_light = LightChop(exp_bits=5, sig_bits=10, rmode=1)
    st = time()
    ch_light(X_th)
    print("runtime 2:", time() - st)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_gpu = X_th.to(device)
    st = time()
    ch(X_gpu)
    print(device, "runtime 3:", time() - st)

    st = time()
    ch_light(X_gpu)
    print(device, "runtime 4:", time() - st)

    # Output
    # python example_gpu.py
    # Load Troch backend.
    # runtime 1: 0.9214024543762207
    # runtime 2: 0.7316868305206299
    # cuda runtime 3: 0.20627665519714355
    # cuda runtime 4: 0.0255887508392334

    # matlab takes: >> tic; emu_val = chop(X); toc -> Elapsed time is 2.084953 seconds. 
