from pychop import LightChop
import jax.numpy as jnp
from jax import random
import numpy as np
import pychop

pychop.backend("jax")

# Initialize with 5 exponent bits and 10 mantissa bits, half precision
ch = LightChop(exp_bits=5, sig_bits=10, rmode=1)

A_np = np.random.randn(5,10)
A_jax = jnp.array(A_np)

# PRNG key for stochastic modes (这里 nearest 不用，但留着也没问题)
key = random.PRNGKey(42)

# Quantize with nearest rounding (no key needed)
result = ch(A_jax)

print(result)
