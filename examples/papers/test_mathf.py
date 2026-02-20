import pychop
from pychop import Chop



import numpy as np
np.random.seed(0)
import pychop.math_func as mf

ch = Chop(exp_bits=5, sig_bits=10, rmode=3)

x = 3.2
X = np.random.randn(10, 10) 
z1 = mf.sin(x, ch)
z2 = mf.sin(X, ch)
z3 = mf.matmul(X, X, ch)
z4 = mf.dot(X, X, ch)

print("====================")
print(z1, z2, z3, z4)