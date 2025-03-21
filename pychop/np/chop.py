import numpy as np
import gc
   

class Chop(object):
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
    Chop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """

    def __init__(self, prec='h', subnormal=None, rmode=1, flip=False, explim=1,
                 p=0.5, randfunc=None, customs=None, random_state=0):
        
        np.random.seed(random_state)
        
        self.prec = prec
        
        if subnormal is not None:
            self.subnormal = subnormal
        else:
            if self.prec in {'b','bfloat16'}:
                self.subnormal = False
            else:
                self.subnormal = True
            
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        
        self.randfunc = randfunc
        # Set rounding function
        self._chop = {
            1: _chop_round_to_nearest,
            2: _chop_round_towards_plus_inf,
            3: _chop_round_towards_minus_inf,
            4: _chop_round_towards_zero,
            5: _chop_stochastic_rounding,
            6: _chop_stochastic_rounding_equal
        }.get(rmode, lambda *args: raise_value_error('Unsupported rmode'))

        # Set precision parameters
        if customs:
            self.t, self.emax = customs.t, customs.emax
        else:
            prec_map = {
                'q43': (4, 7), 'fp8-e4m3': (4, 7), 'q52': (3, 15), 'fp8-e5m2': (3, 15),
                'h': (11, 15), 'half': (11, 15), 'fp16': (11, 15),
                'b': (8, 127), 'bfloat16': (8, 127),
                's': (24, 127), 'single': (24, 127), 'fp32': (24, 127),
                'd': (53, 1023), 'double': (53, 1023), 'fp64': (53, 1023)
            }
            if prec not in prec_map:
                raise ValueError('Invalid prec value')
            self.t, self.emax = prec_map[prec]

        self.u = None
    
            
    def __call__(self, x):
        y = self.chop_wrapper(x.copy())
        return y
        
    def chop_wrapper(self, x):
        if str(x).isnumeric():
            raise ValueError('Chop requires real input values (not int).')
            
        if not hasattr(x, "__len__"):
            x = np.array(x, ndmin=1)
            
        if hasattr(self, 'customs'):
            if self.rmode == 1:
                self.maxfraction = (x.dtype == 'float32')  * 11 + (x.dtype == 'float64')  * 25
            else:
                self.maxfraction = (x.dtype == 'float32')  * 23 + (x.dtype == 'float64') * 52
                
            if self.t > self.maxfraction:
                raise ValueError('Precision of the custom format must be at most')
                
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip, 
                                explim=self.explim, p=self.p)
    

    # Trigonometric Functions
    def sin(self, x):
        x = self.chop_wrapper(x)
        result = np.sin(x)
        return self.chop_wrapper(result)

    def cos(self, x):
        x = self.chop_wrapper(x)
        result = np.cos(x)
        return self.chop_wrapper(result)

    def tan(self, x):
        x = self.chop_wrapper(x)
        result = np.tan(x)
        return self.chop_wrapper(result)

    def arcsin(self, x):
        x = self.chop_wrapper(x)
        if not np.all(np.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        result = np.arcsin(x)
        return self.chop_wrapper(result)

    def arccos(self, x):
        x = self.chop_wrapper(x)
        if not np.all(np.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        result = np.arccos(x)
        return self.chop_wrapper(result)

    def arctan(self, x):
        x = self.chop_wrapper(x)
        result = np.arctan(x)
        return self.chop_wrapper(result)

    # Hyperbolic Functions
    def sinh(self, x):
        
        x = self.chop_wrapper(x)
        result = np.sinh(x)
        return self.chop_wrapper(result)

    def cosh(self, x):
        x = self.chop_wrapper(x)
        result = np.cosh(x)
        return self.chop_wrapper(result)

    def tanh(self, x):
        x = self.chop_wrapper(x)
        result = np.tanh(x)
        return self.chop_wrapper(result)

    def arcsinh(self, x):
        x = self.chop_wrapper(x)
        result = np.arcsinh(x)
        return self.chop_wrapper(result)

    def arccosh(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        result = np.arccosh(x)
        return self.chop_wrapper(result)

    def arctanh(self, x):
        x = self.chop_wrapper(x)
        if not np.all(np.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        result = np.arctanh(x)
        return self.chop_wrapper(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        x = self.chop_wrapper(x)
        result = np.exp(x)
        return self.chop_wrapper(result)

    def expm1(self, x):
        x = self.chop_wrapper(x)
        result = np.expm1(x)
        return self.chop_wrapper(result)

    def log(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x > 0):
            raise ValueError("log input must be positive")
        result = np.log(x)
        return self.chop_wrapper(result)

    def log10(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x > 0):
            raise ValueError("log10 input must be positive")
        result = np.log10(x)
        return self.chop_wrapper(result)

    def log2(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x > 0):
            raise ValueError("log2 input must be positive")
        result = np.log2(x)
        return self.chop_wrapper(result)

    def log1p(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x > -1):
            raise ValueError("log1p input must be > -1")
        result = np.log1p(x)
        return self.chop_wrapper(result)

    # Power and Root Functions
    def sqrt(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        result = np.sqrt(x)
        return self.chop_wrapper(result)

    def cbrt(self, x):
        x = self.chop_wrapper(x)
        result = np.cbrt(x)
        return self.chop_wrapper(result)

    # Miscellaneous Functions
    def abs(self, x):
        x = self.chop_wrapper(x)
        result = np.abs(x)
        return self.chop_wrapper(result)

    def reciprocal(self, x):
        x = self.chop_wrapper(x)
        if not np.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        result = np.reciprocal(x)
        return self.chop_wrapper(result)

    def square(self, x):
        x = self.chop_wrapper(x)
        result = np.square(x)
        return self.chop_wrapper(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        x = self.chop_wrapper(x)
        mantissa, exponent = np.frexp(x)
        return self.chop_wrapper(mantissa), self.chop_wrapper(exponent)

    def hypot(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.hypot(x, y)
        return self.chop_wrapper(result)

    def diff(self, x, n=1):
        x = self.chop_wrapper(x)
        result = np.diff(x, n=n)
        return self.chop_wrapper(result)

    def power(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.power(x, y)
        return self.chop_wrapper(result)

    def modf(self, x):
        x = self.chop_wrapper(x)
        fractional, integer = np.modf(x)
        return self.chop_wrapper(fractional), self.chop_wrapper(integer)

    def ldexp(self, x, i):
        i = np.array(i, dtype=np.int32)  # Exponent not chopped
        x = self.chop_wrapper(x)
        result = np.ldexp(x, i)
        return self.chop_wrapper(result)

    def angle(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.chop_wrapper(x)
        result = np.angle(x)
        return self.chop_wrapper(result)

    def real(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.chop_wrapper(x)
        result = np.real(x)
        return self.chop_wrapper(result)

    def imag(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.chop_wrapper(x)
        result = np.imag(x)
        return self.chop_wrapper(result)

    def conj(self, x):
        x = np.array(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        x = self.chop_wrapper(x)
        result = np.conj(x)
        return self.chop_wrapper(result)

    def maximum(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.maximum(x, y)
        return self.chop_wrapper(result)

    def minimum(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.minimum(x, y)
        return self.chop_wrapper(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.multiply(x, y)
        return self.chop_wrapper(result)

    def mod(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        if not np.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        result = np.mod(x, y)
        return self.chop_wrapper(result)

    def divide(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        if not np.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        result = np.divide(x, y)
        return self.chop_wrapper(result)

    def add(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.add(x, y)
        return self.chop_wrapper(result)

    def subtract(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.subtract(x, y)
        return self.chop_wrapper(result)

    def floor_divide(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        if not np.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        result = np.floor_divide(x, y)
        return self.chop_wrapper(result)

    def bitwise_and(self, x, y):
        
        
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.bitwise_and(x, y)
        return self.chop_wrapper(result)

    def bitwise_or(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.bitwise_or(x, y)
        return self.chop_wrapper(result)

    def bitwise_xor(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.bitwise_xor(x, y)
        return self.chop_wrapper(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = np.sum(x, axis=axis)
        return self.chop_wrapper(result)

    def prod(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = np.prod(x, axis=axis)
        return self.chop_wrapper(result)

    def mean(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = np.mean(x, axis=axis)
        return self.chop_wrapper(result)

    def std(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = np.std(x, axis=axis)
        return self.chop_wrapper(result)

    def var(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = np.var(x, axis=axis)
        return self.chop_wrapper(result)

    def dot(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.dot(x, y)
        return self.chop_wrapper(result)

    def matmul(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.matmul(x, y)
        return self.chop_wrapper(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        x = self.chop_wrapper(x)
        result = np.floor(x)
        return self.chop_wrapper(result)

    def ceil(self, x):
        x = self.chop_wrapper(x)
        result = np.ceil(x)
        return self.chop_wrapper(result)

    def round(self, x, decimals=0):
        x = self.chop_wrapper(x)
        result = np.round(x, decimals=decimals)
        return self.chop_wrapper(result)

    def sign(self, x):
        x = self.chop_wrapper(x)
        result = np.sign(x)
        return self.chop_wrapper(result)

    def clip(self, x, a_min, a_max):
        a_min = np.array(a_min, dtype=np.float64)
        a_max = np.array(a_max, dtype=np.float64)
        x = self.chop_wrapper(x)
        chopped_a_min = self.chop_wrapper(a_min)
        chopped_a_max = self.chop_wrapper(a_max)
        result = np.clip(x, chopped_a_min, chopped_a_max)
        return self.chop_wrapper(result)

    # Special Functions
    def erf(self, x):
        x = self.chop_wrapper(x)
        result = np.special.erf(x)
        return self.chop_wrapper(result)

    def erfc(self, x):
        x = self.chop_wrapper(x)
        result = np.special.erfc(x)
        return self.chop_wrapper(result)

    def gamma(self, x):
        x = self.chop_wrapper(x)
        result = np.special.gamma(x)
        return self.chop_wrapper(result)

    # New Mathematical Functions
    def fabs(self, x):
        """Floating-point absolute value with chopping."""
        
        x = self.chop_wrapper(x)
        result = np.fabs(x)
        return self.chop_wrapper(result)

    def logaddexp(self, x, y):
        """Logarithm of sum of exponentials with chopping."""
        
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = np.logaddexp(x, y)
        return self.chop_wrapper(result)

    def cumsum(self, x, axis=None):
        """Cumulative sum with chopping."""
        
        x = self.chop_wrapper(x)
        result = np.cumsum(x, axis=axis)
        return self.chop_wrapper(result)

    def cumprod(self, x, axis=None):
        """Cumulative product with chopping."""
        
        x = self.chop_wrapper(x)
        result = np.cumprod(x, axis=axis)
        return self.chop_wrapper(result)

    def degrees(self, x):
        """Convert radians to degrees with chopping."""

        x = self.chop_wrapper(x)
        result = np.degrees(x)
        return self.chop_wrapper(result)

    def radians(self, x):
        """Convert degrees to radians with chopping."""

        x = self.chop_wrapper(x)
        result = np.radians(x)
        return self.chop_wrapper(result)

    @property
    def options(self):
        return Options(self.t, 
                       self.emax,
                       self.prec,
                       self.subnormal,
                       self.rmode,
                       self.flip,
                       self.explim,
                       self.p
                      )
    
    
    
def _chop_round_to_nearest(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, *argv, **kwargs):
              
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)

    emin = 1 - emax                # Exponent of smallest normalized number.
    xmin = 2**emin                 # Smallest positive normalized number.
    emins = emin + 1 - t           # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.

    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = np.array([], dtype=bool)
        k_norm = np.full(ktemp.shape, True, dtype=bool)

    w = np.power(2.0, t-1-e[k_norm])
    x[k_norm] = round_to_nearest(
        x=x[k_norm] * w, 
        flip=flip, p=p,
        t=t,
        randfunc=randfunc
    ) 

    x[k_norm] *= 1 / w

    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        x[k_sub] = round_to_nearest(
            x=x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            flip=flip, p=p,
            t=t,
            randfunc=randfunc
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()

    if explim:
        xboundary = 2**emax * (2- 0.5 * 2**(1-t))
        x[x >= xboundary] = np.inf    # Overflow to +inf.
        x[x <= -xboundary] = -np.inf  # Overflow to -inf.
                
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = np.abs(x) < min_rep
        
        if subnormal == 0:
            k_round = k_small & (np.abs(x) >= min_rep/2)
        else:
            k_round = k_small & (np.abs(x) > min_rep/2)
        
        x[k_round] = np.sign(x[k_round]) * min_rep
        x[k_small & ~k_round] = 0

    return x
    
    
    
    

def _chop_round_towards_plus_inf(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, *argv, **kwargs):
              
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)
        
    emin = 1 - emax                # Exponent of smallest normalized number.
    xmin = 2**emin                 # Smallest positive normalized number.
    emins = emin + 1 - t           # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))

    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = np.array([], dtype=bool)
        k_norm = np.full(ktemp.shape, True, dtype=bool)

    w = np.power(2.0, t-1-e[k_norm])
    x[k_norm] = round_towards_plus_inf(
        x=x[k_norm] * w, 
        flip=flip, p=p,
        t=t,
        randfunc=randfunc
    ) 

    x[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        x[k_sub] = round_towards_plus_inf(
            x=x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            flip=flip, p=p,
            t=t,
            randfunc=randfunc
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        x[x > xmax] = np.inf
        x[(x < -xmax) & (x != -np.inf)] = -xmax
                
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = np.abs(x) < min_rep
        
        k_round = k_small & (x > 0) & (x < min_rep)
        x[k_round] = min_rep
        x[k_small & ~k_round] = 0
                
    return x



def _chop_round_towards_minus_inf(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, *argv, **kwargs):
              
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)
        
    emin = 1 - emax                # Exponent of smallest normalized number.
    xmin = 2**emin                 # Smallest positive normalized number.
    emins = emin + 1 - t           # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = np.array([], dtype=bool)
        k_norm = np.full(ktemp.shape, True, dtype=bool)

    w = np.power(2.0, t-1-e[k_norm])
    x[k_norm] = round_towards_minus_inf(
        x=x[k_norm] * w, 
        flip=flip, p=p,
        t=t,
        randfunc=randfunc
    ) 

    x[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        x[k_sub] = round_towards_minus_inf(
            x=x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            flip=flip, p=p,
            t=t,
            randfunc=randfunc
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        x[(x > xmax) & (x != np.inf)] = xmax
        x[x < -xmax] = -np.inf
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = np.abs(x) < min_rep

        k_round = k_small & (x < 0) & (x > -min_rep)
        x[k_round] = -min_rep
        x[k_small & ~k_round] = 0
                
    return x



def _chop_round_towards_zero(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, *argv, **kwargs):
              
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)
        
    emin = 1 - emax                # Exponent of smallest normalized number.
    xmin = 2**emin                 # Smallest positive normalized number.
    emins = emin + 1 - t           # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = np.array([], dtype=bool)
        k_norm = np.full(ktemp.shape, True, dtype=bool)

    w = np.power(2.0, t-1-e[k_norm])
    x[k_norm] = round_towards_zero(
        x=x[k_norm] * w, 
        flip=flip, p=p,
        t=t,
        randfunc=randfunc
    ) 

    x[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        x[k_sub] = round_towards_zero(
            x=x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            flip=flip, p=p,
            t=t,
            randfunc=randfunc
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        x[(x > xmax) & (x != np.inf)] = xmax
        x[(x < -xmax) & (x != -np.inf)] = -xmax
                
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = np.abs(x) < min_rep
        x[k_small] = 0
                
    return x



def _chop_stochastic_rounding(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, *argv, **kwargs):
              
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)
        
    emin = 1 - emax                # Exponent of smallest normalized number.
    xmin = 2**emin                 # Smallest positive normalized number.
    emins = emin + 1 - t           # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = np.array([], dtype=bool)
        k_norm = np.full(ktemp.shape, True, dtype=bool)

    w = np.power(2.0, t-1-e[k_norm])
    x[k_norm] = stochastic_rounding(
        x=x[k_norm] * w, 
        flip=flip, p=p,
        t=t,
        randfunc=randfunc
    ) 

    x[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        x[k_sub] = stochastic_rounding(
            x=x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            flip=flip, p=p,
            t=t,
            randfunc=randfunc
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        x[(x > xmax) & (x != np.inf)] = xmax
        x[(x < -xmax) & (x != -np.inf)] = -xmax
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = np.abs(x) < min_rep
        x[k_small] = 0
                
    return x



def _chop_stochastic_rounding_equal(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, *argv, **kwargs):
              
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)
        
    emin = 1 - emax                # Exponent of smallest normalized number.
    xmin = 2**emin                 # Smallest positive normalized number.
    emins = emin + 1 - t           # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    
    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = np.array([], dtype=bool)
        k_norm = np.full(ktemp.shape, True, dtype=bool)

    w = np.power(2.0, t-1-e[k_norm])
    x[k_norm] = stochastic_rounding_equal(
        x=x[k_norm] * w, 
        flip=flip, p=p,
        t=t,
        randfunc=randfunc
    ) 

    x[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        x[k_sub] = stochastic_rounding_equal(
            x=x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            flip=flip, p=p,
            t=t,
            randfunc=randfunc
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        xboundary = 2**emax * (2- 0.5 * 2**(1-t))
        x[x >= xboundary] = np.inf    # Overflow to +inf.
        x[x <= -xboundary] = -np.inf  # Overflow to -inf.
                
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = np.abs(x) < min_rep
        x[k_small] = 0

    return x


def raise_value_error(msg):
    raise ValueError(msg)







                      
def round_to_nearest(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.abs(x)
    inds = (y - (2 * np.floor(y / 2))) == 0.5
    y[inds] = y[inds] - 1
    u = np.round(y)
    u[u == -1] = 0 # Special case, negative argument to ROUND.
    y = np.sign(x) * u
    
    if flip:
        sign = lambda x: np.sign(x) + (x==0)
        temp = np.random.randint(low=0, high=1, size=y.shape)
        k = temp <= p # Indices of elements to have a bit flipped.
        if np.any(k):
            u = np.abs(y[k])
            b = np.random.randint(low=1, high=t-1, size=u.shape) 
            # Flip selected bits.
            u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
            y[k] = sign(y[k])*u
    
    return y



def round_towards_plus_inf(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.ceil(x)
            
    if flip:
        sign = lambda x: np.sign(x) + (x==0)
        temp = np.random.randint(low=0, high=1, size=y.shape)
        k = temp <= p # Indices of elements to have a bit flipped.
        if np.any(k):
            u = np.abs(y[k])
            b = np.random.randint(low=1, high=t-1, size=u.shape) 
            # Flip selected bits.
            u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
            y[k] = sign(y[k])*u
    
    return y



def round_towards_minus_inf(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.floor(x)
            
    if flip:
        sign = lambda x: np.sign(x) + (x==0)
        temp = np.random.randint(low=0, high=1, size=y.shape)
        k = temp <= p # Indices of elements to have a bit flipped.
        if np.any(k):
            u = np.abs(y[k])
            b = np.random.randint(low=1, high=t-1, size=u.shape) 
            # Flip selected bits.
            u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
            y[k] = sign(y[k])*u
    
    return y


def round_towards_zero(x, flip=0, p=0.5, t=24, **kwargs):
    y = ((x >= 0) | (x == -np.inf)) * np.floor(x) + ((x < 0) | (x == np.inf)) * np.ceil(x)
            
    if flip:
        sign = lambda x: np.sign(x) + (x==0)
        temp = np.random.randint(low=0, high=1, size=y.shape)
        k = temp <= p # Indices of elements to have a bit flipped.
        if np.any(k):
            u = np.abs(y[k])
            b = np.random.randint(low=1, high=t-1, size=u.shape) 
            # Flip selected bits.
            u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
            y[k] = sign(y[k])*u
    
    return y



def stochastic_rounding(x, flip=0, p=0.5, t=24, randfunc=None):
    y = np.abs(x)
    frac = y - np.floor(y)
 
    if np.count_nonzero(frac) == 0:
        y = x 
    else:   
        sign = lambda x: np.sign(x) + (x==0)
        rnd = randfunc(frac.shape)
        j = rnd <= frac
            
        y[j] = np.ceil(y[j])
        y[~j] = np.floor(y[~j])
        y = sign(x)*y
                
        if flip:
            
            temp = np.random.randint(low=0, high=1, size=y.shape)
            k = temp <= p # Indices of elements to have a bit flipped.
            if np.any(k):
                u = np.abs(y[k])
                b = np.random.randint(low=1, high=t-1, size=u.shape) 
                # Flip selected bits.
                u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
                y[k] = sign(y[k])*u
        
    return y



def stochastic_rounding_equal(x, flip=0, p=0.5, t=24, randfunc=None):
    y = np.abs(x)
    frac = y - np.floor(y)
    
    if np.count_nonzero(frac) == 0:
        y = x 
    else:   
        # Uniformly distributed random numbers
        sign = lambda x: np.sign(x) + (x==0)
        rnd = randfunc(frac.shape)
        j = rnd <= 0.5
        y[j] = np.ceil(y[j])
        y[~j] = np.floor(y[~j])
        y = sign(x)*y
            
    if flip:
        sign = lambda x: np.sign(x) + (x==0)
        temp = np.random.randint(low=0, high=1, size=y.shape)
        k = temp <= p # Indices of elements to have a bit flipped.
        if np.any(k):
            u = np.abs(y[k])
            b = np.random.randint(low=1, high=t-1, size=u.shape) 
            # Flip selected bits.
            u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
            y[k] = sign(y[k])*u
    
    return y



def roundit_test(x, rmode=1, flip=0, p=0.5, t=24, randfunc=None):
    if randfunc is None:
        randfunc = lambda n: np.random.randint(0, 1, n)
            

    if rmode == 1:
        y = np.abs(x)
        u = np.round(y - ((y % 2) == 0.5))
        
        u[u == -1] = 0 # Special case, negative argument to ROUND.
            
        y = np.sign(x) * u
        
    elif rmode == 2:
        y = np.ceil(x)
        
    elif rmode == 3:
        y = np.floor(x)
        
    elif rmode == 4:
        y = ((x >= 0) | (x == -np.inf)) * np.floor(x) + ((x < 0) | (x == np.inf)) * np.ceil(x)
        
    elif rmode == 5 | 6:
        y = np.abs(x)
        frac = y - np.floor(y)
        k = np.nonzero(frac != 0)[0]
        
        if len(k) == 0:
            y = x 
        else:   
            # Uniformly distributed random numbers
            
            rnd = randfunc(len(k))
            
            vals = frac[k]
            
            if len(vals.shape) == 2:
                vals = return_column_order(vals)
            else:
                pass
            
            if rmode == 5: # Round up or down with probability prop. to distance.
                j = rnd <= vals
            elif rmode == 6: # Round up or down with equal probability.       
                j = rnd <= 0.5
                
            y[k[j==0]] = np.ceil(y[k[j==0]])
            y[k[j!=0]] = np.floor(y[k[j!=0]])
            y = sign(x)*y
            
    else:
        raise ValueError('Unsupported value of rmode.')
            
    if flip:
        sign = lambda x: np.sign(x) + (x==0)
        temp = np.random.randint(low=0, high=1, size=y.shape)
        k = temp <= p # Indices of elements to have a bit flipped.
        if np.any(k):
            u = np.abs(y[k])
            
            # Random bit flip in significand.
            # b defines which bit (1 to p-1) to flip in each element of y.
            # Using SIZE avoids unwanted implicit expansion.
            # The custom (base 2) format is defined by options.params, which is a
            # 2-vector [t,emax] where t is the number of bits in the significand
            # (including the hidden bit) and emax is the maximum value of the
            # exponent.  

            
            b = np.random.randint(low=1, high=t-1, size=u.shape) 
            # Flip selected bits.
            u = np.bitwise_xor(np.int32(u), np.power(2, b-1))
            y[k] = sign(y[k])*u
    
    return y
    
    
    
   
    
    
    
def return_column_order(arr):
    return arr.T.reshape(-1)
