
import sys
# appending a path
sys.path.append('../')


import numpy as np
import torch
import pychop
from numpy import linalg
from time import time
from pychop.chop import Chop
# from pychop.quant import quant
from time import time

if __name__ == '__main__':
    np.random.seed(0)

    X_np = np.random.randn(7000, 5000) # Numpy array
    X_th = torch.Tensor(X_np) # torch array

    pychop.backend('torch', 1) # print information

    pyq_f = Chop('h')
    st = time()
    pyq_f(X_th)
    print("runtime 1:", time() - st)

    pychop.backend('torch')
    pyq_f = Chop('h')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_gpu = X_th.to(device)

    st = time()
    pyq_f(X_gpu)
    print(device, " runtime 2:", time() - st)

    # Output
    # python example1.py
    # Load Troch backend.
    # runtime 1: 0.9266691207885742
    # cuda  runtime 2: 0.1649622917175293