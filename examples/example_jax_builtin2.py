import pychop
import jax
import jax.numpy as jnp
from pychop import Chop
from pychop.builtin import CPJaxArray

pychop.backend('jax')  # Switch to JAX backend
half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)

x = CPJaxArray(jnp.array([1.1, 2.2, 3.3]), half)
y = CPJaxArray(jnp.array([0.5, 1.5, 2.5]), half)
print(x)                     # CPJaxArray([1.1, 2.2, 3.3], prec=half)
z = x + y
print(z)                     # CPJaxArray([1.6, 3.7, 5.8], prec=half)
# broadcasting with a plain array
reg = jnp.array([10.0, 20.0, 30.0])
w = x * reg
print(w)                     # CPJaxArray([11.0, 44.0, 99.0], prec=half)
# matrix multiplication

gpu_devices = jax.devices('gpu')
if gpu_devices:
    with jax.default_device(gpu_devices[0]):
                    
        A = CPJaxArray(jax.random.normal(jax.random.PRNGKey(0), (4, 3)), half)
        B = CPJaxArray(jax.random.normal(jax.random.PRNGKey(1), (3, 5)), half)
        C = A @ B
        print(C.shape)               # (4, 5)
        # GPU works out-of-the-box (JAX auto-dispatches)
        print(C.to_regular().devices())  # {CpuDevice()} or {CudaDevice()}