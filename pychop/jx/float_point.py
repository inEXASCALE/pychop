"""JAX floating-point chopping backend with stochastic and CADNA rounding."""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np  # Used to initialize the CADNA-compatible generator.


class CADNARandomGeneratorJAX:
    """JAX-compatible CADNA random bit generator.

    The generator follows CADNA's Tausworthe-times-three plus LCG construction
    and returns JAX arrays for use in the JAX backend.
    """
    
    # Tausworthe parameters matching CADNA.
    TAUS1_S1, TAUS1_S2, TAUS1_S3, TAUS1_M = 13, 19, 12, 4294967294
    TAUS2_S1, TAUS2_S2, TAUS2_S3, TAUS2_M = 2, 25, 4, 4294967288
    TAUS3_S1, TAUS3_S2, TAUS3_S3, TAUS3_M = 3, 11, 17, 4294967280
    
    # LCG parameters.
    LCG_A, LCG_C = 1664525, 1013904223
    
    def __init__(self, seed: int = 0):
        """Initialize the generator seeds."""
        np.random.seed(seed)

        # Initialize the four generator seeds with NumPy, then store uint32.
        self.z1 = np.uint32(np.random.randint(128, 2**32 - 1))
        self.z2 = np.uint32(np.random.randint(128, 2**32 - 1))
        self.z3 = np.uint32(np.random.randint(128, 2**32 - 1))
        self.z4 = np.uint32(np.random.randint(1, 2**32 - 1))
        
        self._cache = np.uint32(0)
        self._cache_counter = 0
    
    def _tausworthe_step(self, z: np.uint32, S1: int, S2: int, S3: int, M: int) -> np.uint32:
        """Advance one Tausworthe generator state."""
        b = np.uint32(((z << S1) ^ z) >> S2)
        z = np.uint32(((z & M) << S3) ^ b)
        return z
    
    def _lcg_step(self) -> np.uint32:
        """Advance the LCG state."""
        self.z4 = np.uint32((self.LCG_A * self.z4 + self.LCG_C))
        return self.z4
    
    def _generate_batch(self) -> np.uint32:
        """Generate a batch of 32 random bits."""
        self.z1 = self._tausworthe_step(self.z1, self.TAUS1_S1, self.TAUS1_S2, 
                                         self.TAUS1_S3, self.TAUS1_M)
        self.z2 = self._tausworthe_step(self.z2, self.TAUS2_S1, self.TAUS2_S2, 
                                         self.TAUS2_S3, self.TAUS2_M)
        self.z3 = self._tausworthe_step(self.z3, self.TAUS3_S1, self.TAUS3_S2, 
                                         self.TAUS3_S3, self.TAUS3_M)
        lcg_val = self._lcg_step()
        
        return np.uint32(self.z1 ^ self.z2 ^ self.z3 ^ lcg_val)
    
    def random_bit(self) -> int:
        """Generate one random bit, either 0 or 1."""
        if self._cache_counter % 32 == 0:
            self._cache = self._generate_batch()
            self._cache_counter = 0
        
        bit = int(self._cache & 1)
        self._cache >>= 1
        self._cache_counter += 1
        
        return bit
    
    def random_bits_array(self, shape: tuple) -> jnp.ndarray:
        """Generate random bits as a JAX array.
        
        Parameters
        ----------
        shape : tuple
            Output array shape.
            
        Returns
        -------
        jnp.ndarray
            Random bit array with values 0 or 1 and dtype ``uint8``.
        """
        size = int(np.prod(shape))
        # Generate with NumPy, then convert to a JAX array.
        bits = np.array([self.random_bit() for _ in range(size)], dtype=np.uint8)
        return jnp.array(bits.reshape(shape))


# ============================================================
# JAX sign-bit flipping helpers.
# ============================================================

def jax_bit_flip_float32(x: jnp.ndarray, random_bits: jnp.ndarray) -> jnp.ndarray:
    """Flip the sign bit of float32 values.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array with dtype ``float32``.
    random_bits : jnp.ndarray
        Random bit array with values 0 or 1.
        
    Returns
    -------
    jnp.ndarray
        Array after sign-bit flips.
    """
    # View float32 as uint32 for bit manipulation.
    x_int = x.view(jnp.uint32)
    
    # The sign bit is bit 31.
    sign_bit = jnp.uint32(1) << 31
    
    # XOR flips the sign bit where random_bits is 1.
    mask = random_bits.astype(jnp.uint32) * sign_bit
    x_int_flipped = x_int ^ mask
    
    # View back as float32.
    return x_int_flipped.view(jnp.float32)


def jax_bit_flip_float64(x: jnp.ndarray, random_bits: jnp.ndarray) -> jnp.ndarray:
    """Flip the sign bit of float64 values.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array with dtype ``float64``.
    random_bits : jnp.ndarray
        Random bit array with values 0 or 1.
        
    Returns
    -------
    jnp.ndarray
        Array after sign-bit flips.
    """
    # View float64 as uint64 for bit manipulation.
    x_int = x.view(jnp.uint64)
    
    # The sign bit is bit 63.
    sign_bit = jnp.uint64(1) << 63
    
    # XOR flips the sign bit where random_bits is 1.
    mask = random_bits.astype(jnp.uint64) * sign_bit
    x_int_flipped = x_int ^ mask
    
    # View back as float64.
    return x_int_flipped.view(jnp.float64)


def jax_bit_flip(x: jnp.ndarray, random_bits: jnp.ndarray) -> jnp.ndarray:
    """Flip sign bits using the implementation matching ``x.dtype``.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    random_bits : jnp.ndarray
        Random bit array with values 0 or 1.
        
    Returns
    -------
    jnp.ndarray
        Array after sign-bit flips.
    """
    if x.dtype == jnp.float32:
        return jax_bit_flip_float32(x, random_bits)
    elif x.dtype == jnp.float64:
        return jax_bit_flip_float64(x, random_bits)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")
    

class Chop_(object):
    """
    Parameters
    ----------
    prec : str, default='s':
        The target arithmetic format.
    
    subnormal : boolean
        Whether or not to support subnormal numbers.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int, default=1
        The supported rounding modes include:
        1. Round to nearest using round to even last bit to break ties (the default).
        2. Round towards plus infinity (round up).
        3. Round towards minus infinity (round down).
        4. Round towards zero.
        5. Stochastic rounding - round to the next larger or next smaller
           floating-point number with probability proportional to the distance 
           to those floating-point numbers.
        6. Stochastic rounding - round to the next larger or next smaller 
           floating-point number with equal probability.

        10. Stochastic rounding - CADNA-style random directed rounding.
        
    flip : boolean, default=False
        Default is False; If ``flip`` is True, then each element
        of the rounded result has a randomly generated bit in its significand flipped 
        with probability ``p``. This parameter is designed for soft error simulation. 

    explim : boolean, default=True
        Default is True; If ``explim`` is False, then the maximal exponent for
        the specified arithmetic is ignored, thus overflow, underflow, or subnormal numbers
        will be produced only if necessary for the data type.  
        This option is designed for exploring low precisions independent of range limitations.

    p : float, default=0.5
        The probability ``p` for each element of the rounded result has a randomly
        generated bit in its significand flipped  when ``flip`` is True

    randfunc : callable, default=None
        If ``randfunc`` is supplied, then the random numbers used for rounding  will be generated 
        using that function in stochastic rounding (i.e., ``rmode`` of 5 and 6). Default is numbers
        in uniform distribution between 0 and 1, i.e., np.random.uniform.

    customs : dataclass, default=None
        If customs is defined, then use customs.t and customs.emax for floating point arithmetic.
        where t is the number of bits in the significand (including the hidden bit) and emax
        is the maximum value of the exponent.
    
    random_state : int, default=0
        Random seed set for stochastic rounding settings.

        
    Methods
    ----------
    chop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """
    def __init__(self, prec='h', subnormal=None, rmode=1, flip=False, explim=1,
                 p=0.5, randfunc=None, customs=None, random_state=0):
        self.key = random.PRNGKey(random_state)
        
        self.prec = prec
        self.subnormal = subnormal if subnormal is not None else (prec not in {'b', 'bfloat16'})
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        self.randfunc = randfunc or (lambda key, n: random.uniform(key, (n,)))

        self._chop_funcs = {
            1: _chop_round_to_nearest,
            2: _chop_round_towards_plus_inf,
            3: _chop_round_towards_minus_inf,
            4: _chop_round_towards_zero,
            5: _chop_stochastic_rounding,
            6: _chop_stochastic_rounding_equal,
            10: _chop_cadna_rounding 
        }
        if rmode not in self._chop_funcs:
            raise ValueError('Unsupported value of rmode.')
        self._chop = self._chop_funcs[rmode]

        prec_map = {
            'q43': (4, 7), 'fp8-e4m3': (4, 7), 'q52': (3, 15), 'fp8-e5m2': (3, 15),
            'h': (11, 15), 'half': (11, 15), 'fp16': (11, 15),
            'b': (8, 127), 'bfloat16': (8, 127), 'bf16': (8, 127),
            's': (24, 127), 'single': (24, 127), 'fp32': (24, 127),
            'd': (53, 1023), 'double': (53, 1023), 'fp64': (53, 1023)
        }
        if customs is not None:
            self.t, self.emax = customs.t, customs.emax
        elif prec in prec_map:
            self.t, self.emax = prec_map[prec]
        else:
            raise ValueError('Please enter valid prec value.')
        
        self._emin = 1 - self.emax
        self._xmin = 2.0 ** self._emin
        self._emins = self._emin + 1 - self.t
        self._xmins = 2.0 ** self._emins

        if rmode == 10:
            self._cadna_gen = CADNARandomGeneratorJAX(seed=random_state)
        else:
            self._cadna_gen = None

    def __call__(self, x):
        return self.chop_wrapper(x)

    def chop_wrapper(self, x):
        if isinstance(x, (int, str)) and str(x).isnumeric():
            raise ValueError('Chop requires real input values (not int).')
            
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x, dtype=jnp.float32)
        elif x.dtype in (jnp.int32, jnp.int64):
            x = x.astype(jnp.float32)
            
        if not x.ndim:
            x = x[None]
        
        self.key, subkey = random.split(self.key)
        #return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip, 
        #                 explim=self.explim, p=self.p, key=subkey)
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, 
                        flip=self.flip, explim=self.explim, p=self.p, 
                        key=subkey, cadna_gen=self._cadna_gen)

    # Trigonometric Functions
    def sin(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.sin(x)
        return self.chop_wrapper(result)

    def cos(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.cos(x)
        return self.chop_wrapper(result)

    def tan(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.tan(x)
        return self.chop_wrapper(result)

    def arcsin(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(jnp.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        
        result = jnp.arcsin(x)
        return self.chop_wrapper(result)

    def arccos(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(jnp.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        
        result = jnp.arccos(x)
        return self.chop_wrapper(result)

    def arctan(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.arctan(x)
        return self.chop_wrapper(result)

    # Hyperbolic Functions
    def sinh(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.sinh(x)
        return self.chop_wrapper(result)

    def cosh(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.cosh(x)
        return self.chop_wrapper(result)

    def tanh(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.tanh(x)
        return self.chop_wrapper(result)

    def arcsinh(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.arcsinh(x)
        return self.chop_wrapper(result)

    def arccosh(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        
        result = jnp.arccosh(x)
        return self.chop_wrapper(result)

    def arctanh(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(jnp.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        
        result = jnp.arctanh(x)
        return self.chop_wrapper(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.exp(x)
        return self.chop_wrapper(result)

    def expm1(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.expm1(x)
        return self.chop_wrapper(result)

    def log(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x > 0):
            raise ValueError("log input must be positive")
        
        result = jnp.log(x)
        return self.chop_wrapper(result)

    def log10(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x > 0):
            raise ValueError("log10 input must be positive")
        
        result = jnp.log10(x)
        return self.chop_wrapper(result)

    def log2(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x > 0):
            raise ValueError("log2 input must be positive")
        
        result = jnp.log2(x)
        return self.chop_wrapper(result)

    def log1p(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x > -1):
            raise ValueError("log1p input must be > -1")
        
        result = jnp.log1p(x)
        return self.chop_wrapper(result)

    # Power and Root Functions
    def sqrt(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        
        result = jnp.sqrt(x)
        return self.chop_wrapper(result)

    def cbrt(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.cbrt(x)
        return self.chop_wrapper(result)

    # Miscellaneous Functions
    def abs(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.abs(x)
        return self.chop_wrapper(result)

    def reciprocal(self, x):
        
        x = self.chop_wrapper(x)
        if not jnp.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        
        result = jnp.reciprocal(x)
        return self.chop_wrapper(result)

    def square(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.square(x)
        return self.chop_wrapper(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        
        x = self.chop_wrapper(x)
        mantissa, exponent = jnp.frexp(x)
        
        return self.chop_wrapper(mantissa), exponent  # Exponent typically not chopped

    def hypot(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.hypot(x, y)
        return self.chop_wrapper(result)

    def diff(self, x, n=1):
        
        x = self.chop_wrapper(x)
        for _ in range(n):
            x = jnp.diff(x)
        
        return self.chop_wrapper(x)

    def power(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.power(x, y)
        return self.chop_wrapper(result)

    def modf(self, x):
        
        x = self.chop_wrapper(x)
        fractional, integer = jnp.modf(x)
        
        fractional = self.chop_wrapper(fractional)
        
        integer = self.chop_wrapper(integer)
        return fractional, integer

    def ldexp(self, x, i):
        
        x = self.chop_wrapper(x)
        i = jnp.array(i, dtype=jnp.int32)  # Exponent not chopped
        
        result = jnp.ldexp(x, i)
        return self.chop_wrapper(result)

    def angle(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.angle(x) if jnp.iscomplexobj(x) else jnp.arctan2(x, jnp.zeros_like(x))
        return self.chop_wrapper(result)

    def real(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.real(x) if jnp.iscomplexobj(x) else x
        return self.chop_wrapper(result)

    def imag(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.imag(x) if jnp.iscomplexobj(x) else jnp.zeros_like(x)
        return self.chop_wrapper(result)

    def conj(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.conj(x) if jnp.iscomplexobj(x) else x
        return self.chop_wrapper(result)

    def maximum(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.maximum(x, y)
        return self.chop_wrapper(result)

    def minimum(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.minimum(x, y)
        return self.chop_wrapper(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.multiply(x, y)
        return self.chop_wrapper(result)

    def mod(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        if not jnp.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        
        result = jnp.mod(x, y)
        return self.chop_wrapper(result)

    def divide(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        if not jnp.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        
        result = jnp.divide(x, y)
        return self.chop_wrapper(result)

    def add(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.add(x, y)
        return self.chop_wrapper(result)

    def subtract(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.subtract(x, y)
        return self.chop_wrapper(result)

    def floor_divide(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        if not jnp.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        
        result = jnp.floor_divide(x, y)
        return self.chop_wrapper(result)

    def bitwise_and(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.bitwise_and(x.astype(jnp.int32), y.astype(jnp.int32)).astype(jnp.float32)
        return self.chop_wrapper(result)

    def bitwise_or(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.bitwise_or(x.astype(jnp.int32), y.astype(jnp.int32)).astype(jnp.float32)
        return self.chop_wrapper(result)

    def bitwise_xor(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.bitwise_xor(x.astype(jnp.int32), y.astype(jnp.int32)).astype(jnp.float32)
        return self.chop_wrapper(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.sum(x, axis=axis)
        return self.chop_wrapper(result)

    def prod(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.prod(x, axis=axis)
        return self.chop_wrapper(result)

    def mean(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.mean(x, axis=axis)
        return self.chop_wrapper(result)

    def std(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.std(x, axis=axis)
        return self.chop_wrapper(result)

    def var(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.var(x, axis=axis)
        return self.chop_wrapper(result)

    def dot(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.dot(x, y)
        return self.chop_wrapper(result)

    def matmul(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jnp.matmul(x, y)
        return self.chop_wrapper(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.floor(x)
        return self.chop_wrapper(result)

    def ceil(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.ceil(x)
        return self.chop_wrapper(result)

    def round(self, x, decimals=0):
        
        x = self.chop_wrapper(x)
        if decimals == 0:
            result = jnp.round(x)
        else:
            factor = 10 ** decimals
            result = jnp.round(x * factor) / factor
        
        return self.chop_wrapper(result)

    def sign(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.sign(x)
        return self.chop_wrapper(result)

    def clip(self, x, a_min, a_max):
        
        x = self.chop_wrapper(x)
        a_min = jnp.array(a_min, dtype=jnp.float32)
        a_max = jnp.array(a_max, dtype=jnp.float32)
        
        chopped_a_min = self.chop_wrapper(a_min)
        
        chopped_a_max = self.chop_wrapper(a_max)
        
        result = jnp.clip(x, chopped_a_min, chopped_a_max)
        return self.chop_wrapper(result)

    # Special Functions
    def erf(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jax.scipy.special.erf(x)
        return self.chop_wrapper(result)

    def erfc(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jax.scipy.special.erfc(x)
        return self.chop_wrapper(result)

    def gamma(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jax.scipy.special.gamma(x)
        return self.chop_wrapper(result)

    # New Mathematical Functions
    def fabs(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.fabs(x)
        return self.chop_wrapper(result)

    def logaddexp(self, x, y):
        
        x = self.chop_wrapper(x)
        
        y = self.chop_wrapper(y)
        
        result = jax.scipy.special.logsumexp(jnp.stack([x, y]), axis=0)
        return self.chop_wrapper(result)

    def cumsum(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.cumsum(x, axis=axis)
        return self.chop_wrapper(result)

    def cumprod(self, x, axis=None):
        
        x = self.chop_wrapper(x)
        
        result = jnp.cumprod(x, axis=axis)
        return self.chop_wrapper(result)

    def degrees(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.degrees(x)
        return self.chop_wrapper(result)

    def radians(self, x):
        
        x = self.chop_wrapper(x)
        
        result = jnp.radians(x)
        return self.chop_wrapper(result)
    
    @property
    def options(self):
        return Options(self.t, self.emax, self.prec, self.subnormal, self.rmode, self.flip, self.explim, self.p)


def _chop_round_to_nearest(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, key=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins

    # JAX doesn't have frexp, so use log2 and floor
    abs_x = jnp.abs(x)
    e = jnp.floor(jnp.log2(abs_x)).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)

    w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
    key, subkey = random.split(key)
    x = x.at[k_norm].set(round_to_nearest(x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc, key=subkey))
    x = x.at[k_norm].set(x[k_norm] * (1 / w))

    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        key, subkey = random.split(key)
        x = x.at[k_sub].set(round_to_nearest(x[k_sub] * jnp.power(2, t1 - 1 - e[k_sub].astype(jnp.float32)), 
                                             flip=flip, p=p, t=t, randfunc=randfunc, key=subkey) * 
                            jnp.power(2, e[k_sub].astype(jnp.float32) - (t1 - 1)))

    if explim:
        xboundary = 2 ** emax * (2 - 0.5 * 2 ** (1 - t))
        x = jnp.where(x >= xboundary, jnp.inf, x)
        x = jnp.where(x <= -xboundary, -jnp.inf, x)

        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        k_round = k_small & (jnp.abs(x) > min_rep / 2) if subnormal else k_small & (jnp.abs(x) >= min_rep / 2)
        x = jnp.where(k_round, jnp.sign(x) * min_rep, x)
        x = jnp.where(k_small & ~k_round, 0, x)

    return x

def _chop_round_towards_plus_inf(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, key=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))

    e = jnp.floor(jnp.log2(jnp.abs(x))).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)

    w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
    key, subkey = random.split(key)
    x = x.at[k_norm].set(round_towards_plus_inf(x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc, key=subkey))
    x = x.at[k_norm].set(x[k_norm] * (1 / w))

    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        key, subkey = random.split(key)
        x = x.at[k_sub].set(round_towards_plus_inf(x[k_sub] * jnp.power(2, t1 - 1 - e[k_sub].astype(jnp.float32)), 
                                                  flip=flip, p=p, t=t, randfunc=randfunc, key=subkey) * 
                            jnp.power(2, e[k_sub].astype(jnp.float32) - (t1 - 1)))

    if explim:
        x = jnp.where(x > xmax, jnp.inf, x)
        x = jnp.where((x < -xmax) & (x != -jnp.inf), -xmax, x)
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        k_round = k_small & (x > 0) & (x < min_rep)
        x = jnp.where(k_round, min_rep, x)
        x = jnp.where(k_small & ~k_round, 0, x)

    return x

def _chop_round_towards_minus_inf(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, key=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))

    e = jnp.floor(jnp.log2(jnp.abs(x))).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)

    w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
    key, subkey = random.split(key)
    x = x.at[k_norm].set(round_towards_minus_inf(x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc, key=subkey))
    x = x.at[k_norm].set(x[k_norm] * (1 / w))

    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        key, subkey = random.split(key)
        x = x.at[k_sub].set(round_towards_minus_inf(x[k_sub] * jnp.power(2, t1 - 1 - e[k_sub].astype(jnp.float32)), 
                                                   flip=flip, p=p, t=t, randfunc=randfunc, key=subkey) * 
                            jnp.power(2, e[k_sub].astype(jnp.float32) - (t1 - 1)))

    if explim:
        x = jnp.where((x > xmax) & (x != jnp.inf), xmax, x)
        x = jnp.where(x < -xmax, -jnp.inf, x)
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        k_round = k_small & (x < 0) & (x > -min_rep)
        x = jnp.where(k_round, -min_rep, x)
        x = jnp.where(k_small & ~k_round, 0, x)

    return x

def _chop_round_towards_zero(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, key=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))

    e = jnp.floor(jnp.log2(jnp.abs(x))).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)

    w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
    key, subkey = random.split(key)
    x = x.at[k_norm].set(round_towards_zero(x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc, key=subkey))
    x = x.at[k_norm].set(x[k_norm] * (1 / w))

    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        key, subkey = random.split(key)
        x = x.at[k_sub].set(round_towards_zero(x[k_sub] * jnp.power(2, t1 - 1 - e[k_sub].astype(jnp.float32)), 
                                              flip=flip, p=p, t=t, randfunc=randfunc, key=subkey) * 
                            jnp.power(2, e[k_sub].astype(jnp.float32) - (t1 - 1)))

    if explim:
        x = jnp.where((x > xmax) & (x != jnp.inf), xmax, x)
        x = jnp.where((x < -xmax) & (x != -jnp.inf), -xmax, x)
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        x = jnp.where(k_small, 0, x)

    return x

def _chop_stochastic_rounding(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, key=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))

    e = jnp.floor(jnp.log2(jnp.abs(x))).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)

    w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
    key, subkey = random.split(key)
    x = x.at[k_norm].set(stochastic_rounding(x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc, key=subkey))
    x = x.at[k_norm].set(x[k_norm] * (1 / w))

    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        key, subkey = random.split(key)
        x = x.at[k_sub].set(stochastic_rounding(x[k_sub] * jnp.power(2, t1 - 1 - e[k_sub].astype(jnp.float32)), 
                                               flip=flip, p=p, t=t, randfunc=randfunc, key=subkey) * 
                            jnp.power(2, e[k_sub].astype(jnp.float32) - (t1 - 1)))

    if explim:
        x = jnp.where((x > xmax) & (x != jnp.inf), xmax, x)
        x = jnp.where((x < -xmax) & (x != -jnp.inf), -xmax, x)
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        x = jnp.where(k_small, 0, x)

    return x

def _chop_stochastic_rounding_equal(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, key=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins

    e = jnp.floor(jnp.log2(jnp.abs(x))).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)

    w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
    key, subkey = random.split(key)
    x = x.at[k_norm].set(stochastic_rounding_equal(x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc, key=subkey))
    x = x.at[k_norm].set(x[k_norm] * (1 / w))

    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        key, subkey = random.split(key)
        x = x.at[k_sub].set(stochastic_rounding_equal(x[k_sub] * jnp.power(2, t1 - 1 - e[k_sub].astype(jnp.float32)), 
                                                     flip=flip, p=p, t=t, randfunc=randfunc, key=subkey) * 
                            jnp.power(2, e[k_sub].astype(jnp.float32) - (t1 - 1)))

    if explim:
        xboundary = 2 ** emax * (2 - 0.5 * 2 ** (1 - t))
        x = jnp.where(x >= xboundary, jnp.inf, x)
        x = jnp.where(x <= -xboundary, -jnp.inf, x)
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        x = jnp.where(k_small, 0, x)

    return x


def _chop_cadna_rounding(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5,
                         randfunc=None, key=None, cadna_gen=None, *argv, **kwargs):
    """Apply CADNA-style random directed rounding to a JAX array.

    The mantissa is rounded by randomly choosing the upward or downward
    directed result. The downward branch is implemented with the CADNA sign-bit
    flip trick.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    t : int
        Significand precision, including the hidden bit.
    emax : int
        Maximum exponent.
    subnormal : int
        Whether subnormal numbers are supported.
    flip : int
        Whether to apply the optional soft-error bit flip.
    explim : int
        Whether to apply exponent limits.
    p : float
        Bit-flip probability when ``flip`` is enabled.
    randfunc : callable
        Random function kept for API compatibility.
    key : jax.random.PRNGKey
        JAX random key.
    cadna_gen : CADNARandomGeneratorJAX
        CADNA-style random bit generator.
        
    Returns
    -------
    jnp.ndarray
        Rounded array.
    """
    if cadna_gen is None:
        cadna_gen = CADNARandomGeneratorJAX()
    
    if key is None:
        key = random.PRNGKey(0)
    
    # Precompute constants.
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))
    
    # Compute exponents.
    abs_x = jnp.abs(x)
    e = jnp.floor(jnp.log2(abs_x)).astype(jnp.int32)
    ktemp = (e < emin) & (e >= emins)
    
    # Determine normal and subnormal regions.
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = jnp.zeros_like(ktemp, dtype=jnp.bool_)
        k_norm = jnp.ones_like(ktemp, dtype=jnp.bool_)
    
    # Handle the normal range.
    if jnp.any(k_norm):
        w = jnp.power(2.0, t - 1 - e[k_norm].astype(jnp.float32))
        x_norm = x[k_norm] * w
        
        # Apply CADNA-style rounding.
        x_norm_rounded = cadna_style_rounding(x_norm, flip=flip, p=p, t=t, 
                                              randfunc=randfunc, key=key, 
                                              cadna_gen=cadna_gen)
        x_norm_rounded = x_norm_rounded * (1 / w)
        
        # Update the immutable JAX array.
        x = x.at[k_norm].set(x_norm_rounded)
    
    # Handle the subnormal range.
    if jnp.any(k_sub):
        temp = emin - e[k_sub]
        t1 = t - jnp.maximum(temp, jnp.zeros_like(temp))
        
        w_sub = jnp.power(2.0, t1 - 1 - e[k_sub].astype(jnp.float32))
        x_sub = x[k_sub] * w_sub
        
        # Apply CADNA-style rounding.
        key, subkey = random.split(key)
        x_sub_rounded = cadna_style_rounding(x_sub, flip=flip, p=p, t=t, 
                                            randfunc=randfunc, key=subkey, 
                                            cadna_gen=cadna_gen)
        x_sub_rounded = x_sub_rounded * jnp.power(2.0, e[k_sub].astype(jnp.float32) - (t1 - 1))
        
        # Update the array.
        x = x.at[k_sub].set(x_sub_rounded)
    
    # Boundary handling.
    if explim:
        # Overflow handling.
        x = jnp.where((x > xmax) & (x != jnp.inf), xmax, x)
        x = jnp.where((x < -xmax) & (x != -jnp.inf), -xmax, x)
        
        # Underflow handling.
        min_rep = xmin if subnormal == 0 else xmins
        k_small = jnp.abs(x) < min_rep
        x = jnp.where(k_small, 0, x)
    
    return x


def round_to_nearest(x, flip=0, p=0.5, t=24, randfunc=None, key=None, **kwargs):
    """Round scaled mantissas to nearest with ties to even.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    y = jnp.abs(x)
    inds = (y - (2 * jnp.floor(y / 2))) == 0.5
    y = y.at[inds].set(y[inds] - 1)
    u = jnp.round(y)
    u = u.at[u == -1].set(0)  # Special case
    y = jnp.sign(x) * u
    
    if flip:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y

def round_towards_plus_inf(x, flip=0, p=0.5, t=24, randfunc=None, key=None, **kwargs):
    """Round scaled mantissas toward positive infinity.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    y = jnp.ceil(x)
    
    if flip:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y

def round_towards_minus_inf(x, flip=0, p=0.5, t=24, randfunc=None, key=None, **kwargs):
    """Round scaled mantissas toward negative infinity.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    y = jnp.floor(x)
    
    if flip:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y

def round_towards_zero(x, flip=0, p=0.5, t=24, randfunc=None, key=None, **kwargs):
    """Round scaled mantissas toward zero.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    y = ((x >= 0) | (x == -jnp.inf)) * jnp.floor(x) + ((x < 0) | (x == jnp.inf)) * jnp.ceil(x)
    
    if flip:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y

def stochastic_rounding(x, flip=0, p=0.5, t=24, randfunc=None, key=None):
    """Stochastically round scaled mantissas with distance-proportional odds.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.
    key : jax.random.PRNGKey, default=None
        Random key for stochastic rounding and optional bit flips.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    y = jnp.abs(x)
    frac = y - jnp.floor(y)
    
    if not jnp.any(frac):
        y = x
    else:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        rnd = randfunc(x.size, subkey).reshape(x.shape)
        j = rnd <= frac
        y = jnp.where(j, jnp.ceil(y), jnp.floor(y))
        y = sign(x) * y
        
        if flip:
            key, subkey = random.split(key)
            temp = random.randint(subkey, x.shape, 0, 2)
            k = temp <= p
            if jnp.any(k):
                u = jnp.abs(y[k])
                key, subkey = random.split(key)
                b = random.randint(subkey, u.shape, 1, t - 1)
                u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
                y = y.at[k].set(sign(y[k]) * u)
    
    return y

def stochastic_rounding_equal(x, flip=0, p=0.5, t=24, randfunc=None, key=None):
    """Stochastically round scaled mantissas with equal up/down odds.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.
    key : jax.random.PRNGKey, default=None
        Random key for stochastic rounding and optional bit flips.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.uniform(key, (n,))
    if key is None:
        key = random.PRNGKey(0)

    y = jnp.abs(x)
    frac = y - jnp.floor(y)
    
    if not jnp.any(frac):
        y = x
    else:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        rnd = randfunc(x.size, subkey).reshape(x.shape)
        j = rnd <= 0.5
        y = jnp.where(j, jnp.ceil(y), jnp.floor(y))
        y = sign(x) * y
    
    if flip:
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y


# ============================================================
# CADNA-style Random Rounding (rmode=10)
# ============================================================

def cadna_style_rounding(x, flip=0, p=0.5, t=24, randfunc=None, key=None, cadna_gen=None):
    """Round scaled mantissas with CADNA-style random directed rounding.

    A random bit chooses upward rounding or simulated downward rounding through
    a sign-bit flip.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    flip : int
        Whether to apply the optional soft-error bit flip.
    p : float
        Bit-flip probability when ``flip`` is enabled.
    t : int
        Significand precision.
    randfunc : callable
        Random function kept for API compatibility.
    key : jax.random.PRNGKey
        JAX random key used by optional bit flips.
    cadna_gen : CADNARandomGeneratorJAX
        CADNA-style random bit generator.
        
    Returns
    -------
    jnp.ndarray
        Rounded mantissas.
    """
    if cadna_gen is None:
        cadna_gen = CADNARandomGeneratorJAX()
    
    random_bits = cadna_gen.random_bits_array(x.shape).astype(jnp.bool_)

    # CADNA random directed rounding chooses either predecessor or successor
    # with equal probability in the scaled integer domain.
    y = jnp.where(random_bits, jnp.floor(x), jnp.ceil(x))
    y = jnp.where(jnp.isfinite(x), y, x)

    # Optional soft-error bit flipping.
    if flip:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        if key is None:
            key = random.PRNGKey(0)
        
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), 
                               jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y


def roundit_test(x, rmode=1, flip=0, p=0.5, t=24, randfunc=None, key=None):
    """Apply a standalone rounding mode for implementation checks.

    Parameters
    ----------
    x : jax.Array
        Scaled mantissa values.
    rmode : int, default=1
        Rounding mode to apply.
    key : jax.random.PRNGKey, default=None
        Random key used by stochastic modes and optional bit flips.

    Returns
    -------
    jax.Array
        Rounded mantissas.
    """
    if randfunc is None:
        randfunc = lambda n, key: random.randint(key, (n,), 0, 2)
    if key is None:
        key = random.PRNGKey(0)

    if rmode == 1:
        y = jnp.abs(x)
        u = jnp.round(y - ((y % 2) == 0.5).astype(jnp.float32))
        u = u.at[u == -1].set(0)
        y = jnp.sign(x) * u
    elif rmode == 2:
        y = jnp.ceil(x)
    elif rmode == 3:
        y = jnp.floor(x)
    elif rmode == 4:
        y = ((x >= 0) | (x == -jnp.inf)) * jnp.floor(x) + ((x < 0) | (x == jnp.inf)) * jnp.ceil(x)
    elif rmode in (5, 6):
        y = jnp.abs(x)
        frac = y - jnp.floor(y)
        k = jnp.nonzero(frac != 0, size=x.size)[0]
        
        if k.size == 0:
            y = x
        else:
            key, subkey = random.split(key)
            rnd = randfunc(k.size, subkey)
            vals = frac[k]
            
            if rmode == 5:
                j = rnd <= vals
            elif rmode == 6:
                j = rnd <= 0.5
                
            y = y.at[k[j == 0]].set(jnp.ceil(y[k[j == 0]]))
            y = y.at[k[j != 0]].set(jnp.floor(y[k[j != 0]]))
            y = jnp.sign(x) * y
    else:
        raise ValueError('Unsupported value of rmode.')
    
    if flip:
        sign = lambda x: jnp.sign(x) + (x == 0).astype(jnp.float32)
        key, subkey = random.split(key)
        temp = random.randint(subkey, x.shape, 0, 2)
        k = temp <= p
        if jnp.any(k):
            u = jnp.abs(y[k])
            key, subkey = random.split(key)
            b = random.randint(subkey, u.shape, 1, t - 1)
            u = jnp.bitwise_xor(u.astype(jnp.int32), jnp.power(2, b - 1).astype(jnp.int32)).astype(jnp.float32)
            y = y.at[k].set(sign(y[k]) * u)
    
    return y

def return_column_order(arr):
    """Return a flattened copy in column-major traversal order.

    Parameters
    ----------
    arr : jax.Array
        Input array.

    Returns
    -------
    jax.Array
        Flattened array using column-major ordering.
    """
    return arr.T.reshape(-1)
