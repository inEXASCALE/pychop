import pychop
from pychop import Chop


import numpy as np
import pychop.math_func as mf


chopper = Chop(exp_bits=5, sig_bits=10, rmode=3)
x = np.array([0.0, 1.5708])  # ~ [0, pi/2]
result = mf.sin(x, chopper)
print(result)  # Expected: ~ [0.0, 1.0] with chopping

import torch
import pychop.math_func as mf


chopper = Chop(exp_bits=5, sig_bits=10, rmode=3)
x = torch.tensor([0.0, 1.5708])
result = mf.sin(x, chopper)
print(result)  # Expected: ~ [0.0, 1.0] with chopping


import jax.numpy as jnp
import pychop.math_func as mf


chopper = Chop(exp_bits=5, sig_bits=10, rmode=3)
x = jnp.array([0.0, 1.5708])
result = mf.sin(x, chopper)
print(result)  # Expected: ~ [0.0, 1.0] with chopping