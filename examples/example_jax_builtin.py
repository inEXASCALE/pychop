import pychop
import jax
import jax.numpy as jnp
from pychop import Chop
from pychop.builtin import CPJaxArray

pychop.backend('jax')  # Switch to JAX backend

# ── Precision configs─────
half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)

# ── Basic arithmetic─────
x = CPJaxArray(jnp.array([1.1, 2.2, 3.3]), half)
y = CPJaxArray(jnp.array([0.5, 1.5, 2.5]), half)
print("x =", x)
print("y =", y)

z = x + y
print("x + y =", z)

z = x - y
print("x - y =", z)

z = x * y
print("x * y =", z)

z = x / y
print("x / y =", z)

# ── Unary ops ──
print("-x =", -x)
print("|x| =", abs(-x))

# ── Broadcasting with a plain JAX array 
reg = jnp.array([10.0, 20.0, 30.0])
w = x * reg
print("x * plain =", w)

# ── Reverse ops (plain on left, CPJaxArray on right) 
w2 = reg + x
print("plain + x =", w2)          # Should now be CPJaxArray!

w3 = reg - x
print("plain - x =", w3)

w4 = reg * x
print("plain * x =", w4)

# ── Matrix multiplication
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
A = CPJaxArray(jax.random.normal(k1, (4, 3)), half)
B = CPJaxArray(jax.random.normal(k2, (3, 5)), half)
C = A @ B
print("A @ B shape =", C.shape)
print("A @ B =", C)

plain_mat = jnp.ones((2, 4))
D = plain_mat @ A
print("plain @ A =", D)

# ── Comparisons 
print("x > y =", x > y)
print("x == x =", x == x)

# ── Indexing ────
print("x[0] =", x[0])              # Scalar
print("x[1:] =", x[1:])            # Slice -> CPJaxArray

# ── to_regular (unwrap to plain jax.Array) ─────────────────────────
plain = x.to_regular()
print("to_regular type =", type(plain))
print("to_regular value =", plain)

# ── jit compatibility────
@jax.jit
def add_chopped(a, b):
    return a + b

try:
    result = add_chopped(x, y)
    print("jit(x + y) =", result)
except Exception as e:
    print(f"jit not supported (Chop may need jnp internals): {e}")

# ── vmap compatibility───
# Note: vmap requires Chop.__call__ to handle batched tracers.
# If Chop uses scalar Python ops internally, this will fail.
@jax.vmap
def scale(a):
    return a * 2.0

try:
    result = scale(x)
    print("vmap(x * 2) =", result)
except Exception as e:
    print(f"vmap not supported (expected if Chop uses non-jnp ops): {type(e).__name__}")

try:
    bad = CPJaxArray(jnp.array([1.0]), ufhalf)
    _ = x + bad
except ValueError as e:
    print(f"Mixed chopper error (expected): {e}")

# ── GPU support ─
try:
    gpu_devices = jax.devices('gpu')
    if gpu_devices:
        with jax.default_device(gpu_devices[0]):
            A_gpu = CPJaxArray(jax.random.normal(k1, (4, 3)), half)
            B_gpu = CPJaxArray(jax.random.normal(k2, (3, 5)), half)
            C_gpu = A_gpu @ B_gpu
            print("GPU device =", C_gpu.to_regular().devices())
except RuntimeError:
    print("No GPU available, skipping GPU test")
