import sys
# appending a path
sys.path.append('../')

from pychop.chop import chop
import pychop

from scipy.io import loadmat
from time import time

# pychop.backend('torch')
pychop.backend('numpy', 1) # print information, NumPy is the default option.
X_np = loadmat("verified_data.mat")
X_np = X_np['array'][0]

ch = chop('h')
st = time()
X_bit = ch(X_np)
print("runtime:", time() - st)
print(X_bit[:10])