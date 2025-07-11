import torch
from .. import Options

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
        torch.manual_seed(random_state)
        
        self.prec = prec
        self.subnormal = subnormal if subnormal is not None else (prec not in {'b', 'bfloat16'})
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        self.randfunc = randfunc or (lambda n, device: torch.rand(n, device=device))

        self._chop_funcs = {
            1: _chop_round_to_nearest,
            2: _chop_round_towards_plus_inf,
            3: _chop_round_towards_minus_inf,
            4: _chop_round_towards_zero,
            5: _chop_stochastic_rounding,
            6: _chop_stochastic_rounding_equal
        }
        if rmode not in self._chop_funcs:
            raise ValueError('Unsupported value of rmode.')
        self._chop = self._chop_funcs[rmode]

        prec_map = {
            'q43': (4, 7), 'fp8-e4m3': (4, 7), 'q52': (3, 15), 'fp8-e5m2': (3, 15),
            'h': (11, 15), 'half': (11, 15), 'fp16': (11, 15),
            'b': (8, 127), 'bfloat16': (8, 127),
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
        self._xmin = torch.tensor(2.0 ** self._emin, dtype=torch.float32)
        self._emins = self._emin + 1 - self.t
        self._xmins = torch.tensor(2.0 ** self._emins, dtype=torch.float32)

    def __call__(self, x):
        return self.chop_wrapper(x.clone())

    def chop_wrapper(self, x):
        if isinstance(x, (int, str)) and str(x).isnumeric():
            raise ValueError('Chop requires real input values (not int).')
            
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype in (torch.int32, torch.int64):
            x = x.to(torch.float32)
            
        if not x.ndim:
            x = x.unsqueeze(0)
            
        # Move class constants to the same device as x
        self._xmin = self._xmin.to(x.device)
        self._xmins = self._xmins.to(x.device)
            
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip, 
                         explim=self.explim, p=self.p, randfunc=lambda n: self.randfunc(n, device=x.device))



    # Trigonometric Functions
    def sin(self, x):
        x = self.chop_wrapper(x)
        result = torch.sin(x)
        return self.chop_wrapper(result)

    def cos(self, x):
        x = self.chop_wrapper(x)
        result = torch.cos(x)
        return self.chop_wrapper(result)

    def tan(self, x):
        x = self.chop_wrapper(x)
        result = torch.tan(x)
        return self.chop_wrapper(result)

    def arcsin(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        result = torch.asin(x)
        return self.chop_wrapper(result)

    def arccos(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        result = torch.acos(x)
        return self.chop_wrapper(result)

    def arctan(self, x):
        x = self.chop_wrapper(x)
        result = torch.atan(x)
        return self.chop_wrapper(result)

    # Hyperbolic Functions
    def sinh(self, x):
        x = self.chop_wrapper(x)
        result = torch.sinh(x)
        return self.chop_wrapper(result)

    def cosh(self, x):
        x = self.chop_wrapper(x)
        result = torch.cosh(x)
        return self.chop_wrapper(result)

    def tanh(self, x):
        x = self.chop_wrapper(x)
        result = torch.tanh(x)
        return self.chop_wrapper(result)

    def arcsinh(self, x):
        x = self.chop_wrapper(x)
        result = torch.asinh(x)
        return self.chop_wrapper(result)

    def arccosh(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        result = torch.acosh(x)
        return self.chop_wrapper(result)

    def arctanh(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(torch.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        result = torch.atanh(x)
        return self.chop_wrapper(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        x = self.chop_wrapper(x)
        result = torch.exp(x)
        return self.chop_wrapper(result)

    def expm1(self, x):
        x = self.chop_wrapper(x)
        result = torch.expm1(x)
        return self.chop_wrapper(result)

    def log(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x > 0):
            raise ValueError("log input must be positive")
        result = torch.log(x)
        return self.chop_wrapper(result)

    def log10(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x > 0):
            raise ValueError("log10 input must be positive")
        result = torch.log10(x)
        return self.chop_wrapper(result)

    def log2(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x > 0):
            raise ValueError("log2 input must be positive")
        result = torch.log2(x)
        return self.chop_wrapper(result)

    def log1p(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x > -1):
            raise ValueError("log1p input must be > -1")
        result = torch.log1p(x)
        return self.chop_wrapper(result)

    # Power and Root Functions
    def sqrt(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        result = torch.sqrt(x)
        return self.chop_wrapper(result)

    def cbrt(self, x):
        x = self.chop_wrapper(x)
        result = torch.pow(x, 1/3)
        return self.chop_wrapper(result)

    # Miscellaneous Functions
    def abs(self, x):
        x = self.chop_wrapper(x)
        result = torch.abs(x)
        return self.chop_wrapper(result)

    def reciprocal(self, x):
        x = self.chop_wrapper(x)
        if not torch.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        result = torch.reciprocal(x)
        return self.chop_wrapper(result)

    def square(self, x):
        x = self.chop_wrapper(x)
        result = torch.square(x)
        return self.chop_wrapper(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        x = self.chop_wrapper(x)
        mantissa, exponent = torch.frexp(x)
        return self.chop_wrapper(mantissa), exponent  # Exponent typically not chopped

    def hypot(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.hypot(x, y)
        return self.chop_wrapper(result)

    def diff(self, x, n=1):
        x = self.chop_wrapper(x)
        for _ in range(n):
            x = torch.diff(x)
        return self.chop_wrapper(x)

    def power(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.pow(x, y)
        return self.chop_wrapper(result)

    def modf(self, x):
        x = self.chop_wrapper(x)
        fractional, integer = torch.modf(x)
        return self.chop_wrapper(fractional), self.chop_wrapper(integer)

    def ldexp(self, x, i):
        x = self.chop_wrapper(x)
        i = torch.tensor(i, dtype=torch.int32, device=x.device)  # Exponent not chopped
        result = x * torch.pow(2.0, i)
        return self.chop_wrapper(result)

    def angle(self, x):
        if torch.is_complex(x):
            x = self.chop_wrapper(x)
            result = torch.angle(x)
        else:
            x = self.chop_wrapper(x)
            result = torch.atan2(x, torch.zeros_like(x))
        return self.chop_wrapper(result)

    def real(self, x):
        x = self.chop_wrapper(x)
        result = torch.real(x) if torch.is_complex(x) else x
        return self.chop_wrapper(result)

    def imag(self, x):
        x = self.chop_wrapper(x)
        result = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(x)
        return self.chop_wrapper(result)

    def conj(self, x):
        x = self.chop_wrapper(x)
        result = torch.conj(x) if torch.is_complex(x) else x
        return self.chop_wrapper(result)

    def maximum(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.maximum(x, y)
        return self.chop_wrapper(result)

    def minimum(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.minimum(x, y)
        return self.chop_wrapper(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.mul(x, y)
        return self.chop_wrapper(result)

    def mod(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        if not torch.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        result = torch.fmod(x, y)
        return self.chop_wrapper(result)

    def divide(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        if not torch.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        result = torch.div(x, y)
        return self.chop_wrapper(result)

    def add(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.add(x, y)
        return self.chop_wrapper(result)

    def subtract(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.sub(x, y)
        return self.chop_wrapper(result)

    def floor_divide(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        if not torch.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        result = torch.div(x, y, rounding_mode='floor')
        return self.chop_wrapper(result)

    def bitwise_and(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.bitwise_and(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.chop_wrapper(result)

    def bitwise_or(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.bitwise_or(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.chop_wrapper(result)

    def bitwise_xor(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.bitwise_xor(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.chop_wrapper(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.sum(x, dim=axis)
        return self.chop_wrapper(result)

    def prod(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.prod(x, dim=axis)
        return self.chop_wrapper(result)

    def mean(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.mean(x, dim=axis)
        return self.chop_wrapper(result)

    def std(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.std(x, dim=axis)
        return self.chop_wrapper(result)

    def var(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.var(x, dim=axis)
        return self.chop_wrapper(result)

    def dot(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.dot(x, y)
        return self.chop_wrapper(result)

    def matmul(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.matmul(x, y)
        return self.chop_wrapper(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        x = self.chop_wrapper(x)
        result = torch.floor(x)
        return self.chop_wrapper(result)

    def ceil(self, x):
        x = self.chop_wrapper(x)
        result = torch.ceil(x)
        return self.chop_wrapper(result)

    def round(self, x, decimals=0):
        x = self.chop_wrapper(x)
        if decimals == 0:
            result = torch.round(x)
        else:
            factor = 10 ** decimals
            result = torch.round(x * factor) / factor
        return self.chop_wrapper(result)

    def sign(self, x):
        x = self.chop_wrapper(x)
        result = torch.sign(x)
        return self.chop_wrapper(result)

    def clip(self, x, a_min, a_max):
        a_min = torch.tensor(a_min, dtype=torch.float32, device=x.device)
        a_max = torch.tensor(a_max, dtype=torch.float32, device=x.device)
        x = self.chop_wrapper(x)
        chopped_a_min = self.chop_wrapper(a_min)
        chopped_a_max = self.chop_wrapper(a_max)
        result = torch.clamp(x, min=chopped_a_min, max=chopped_a_max)
        return self.chop_wrapper(result)

    # Special Functions
    def erf(self, x):
        x = self.chop_wrapper(x)
        result = torch.erf(x)
        return self.chop_wrapper(result)

    def erfc(self, x):
        x = self.chop_wrapper(x)
        result = torch.erfc(x)
        return self.chop_wrapper(result)

    def gamma(self, x):
        x = self.chop_wrapper(x)
        result = torch.special.gamma(x)
        return self.chop_wrapper(result)

    # New Mathematical Functions
    def fabs(self, x):
        x = self.chop_wrapper(x)
        result = torch.abs(x)
        return self.chop_wrapper(result)

    def logaddexp(self, x, y):
        x = self.chop_wrapper(x)
        y = self.chop_wrapper(y)
        result = torch.logaddexp(x, y)
        return self.chop_wrapper(result)

    def cumsum(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.cumsum(x, dim=axis)
        return self.chop_wrapper(result)

    def cumprod(self, x, axis=None):
        x = self.chop_wrapper(x)
        result = torch.cumprod(x, dim=axis)
        return self.chop_wrapper(result)

    def degrees(self, x):
        x = self.chop_wrapper(x)
        result = torch.deg2rad(x) * (180 / torch.pi)
        return self.chop_wrapper(result)

    def radians(self, x):
        x = self.chop_wrapper(x)
        result = torch.rad2deg(x) * (torch.pi / 180)
        return self.chop_wrapper(result)
    

    @property
    def options(self):
        return Options(self.t, self.emax, self.prec, self.subnormal, self.rmode, self.flip, self.explim, self.p)



def _chop_round_to_nearest(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins

    _, e = torch.frexp(torch.abs(x))
    e = e - 1
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool, device=x.device)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool, device=x.device)

    w = torch.pow(2.0, t - 1 - e[k_norm].float()).to(x.device)
    x[k_norm] = round_to_nearest(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w

    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp, device=x.device))
        x[k_sub] = round_to_nearest(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                    flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        xboundary = 2 ** emax * (2 - 0.5 * 2 ** (1 - t))
        x[x >= xboundary] = float('inf')
        x[x <= -xboundary] = float('-inf')

        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        k_round = k_small & (torch.abs(x) > min_rep / 2) if subnormal else k_small & (torch.abs(x) >= min_rep / 2)
        
        x[k_round] = torch.sign(x[k_round]) * min_rep
        x[k_small & ~k_round] = 0

    return x

def _chop_round_towards_plus_inf(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
        
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))

    _, e = torch.frexp(torch.abs(x))
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool, device=x.device)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool, device=x.device)

    w = torch.pow(2.0, t - 1 - e[k_norm].float()).to(x.device)
    x[k_norm] = round_towards_plus_inf(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp, device=x.device))
        x[k_sub] = round_towards_plus_inf(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                          flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        x[x > xmax] = float('inf')
        x[(x < -xmax) & (x != float('-inf'))] = -xmax
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        k_round = k_small & (x > 0) & (x < min_rep)
        x[k_round] = min_rep
        x[k_small & ~k_round] = 0
                
    return x

def _chop_round_towards_minus_inf(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
        
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))
    
    _, e = torch.frexp(torch.abs(x))
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool, device=x.device)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool, device=x.device)

    w = torch.pow(2.0, t - 1 - e[k_norm].float()).to(x.device)
    x[k_norm] = round_towards_minus_inf(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp, device=x.device))
        x[k_sub] = round_towards_minus_inf(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                           flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        x[(x > xmax) & (x != float('inf'))] = xmax
        x[x < -xmax] = float('-inf')
        
        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        k_round = k_small & (x < 0) & (x > -min_rep)
        x[k_round] = -min_rep
        x[k_small & ~k_round] = 0
                
    return x

def _chop_round_towards_zero(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
        
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))
    
    _, e = torch.frexp(torch.abs(x))
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool, device=x.device)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool, device=x.device)

    w = torch.pow(2.0, t - 1 - e[k_norm].float()).to(x.device)
    x[k_norm] = round_towards_zero(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp, device=x.device))
        x[k_sub] = round_towards_zero(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                      flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        x[(x > xmax) & (x != float('inf'))] = xmax
        x[(x < -xmax) & (x != float('-inf'))] = -xmax
        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        x[k_small] = 0
                
    return x

# Optimized versions with consistent output
def _chop_stochastic_rounding(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)

    # Precompute constants
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))

    # Efficient exponent calculation
    abs_x = torch.abs(x)
    _, e = torch.frexp(abs_x)
    e = e - 1
    ktemp = (e < emin) & (e >= emins)

    # Minimize tensor creation
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.empty_like(ktemp, dtype=torch.bool, device=x.device).fill_(False)
        k_norm = torch.empty_like(ktemp, dtype=torch.bool, device=x.device).fill_(True)

    # Normal range: avoid in-place to match original
    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x_norm = x[k_norm] * w
    x_norm = stochastic_rounding(x_norm, flip=flip, p=p, t=t, randfunc=randfunc) * (1 / w)
    x[k_norm] = x_norm

    # Subnormal range: avoid in-place to match original
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.clamp(temp, min=0)  # Optimized with clamp
        w_sub = torch.pow(2.0, t1 - 1 - e[k_sub].float())
        x_sub = x[k_sub] * w_sub
        x_sub = stochastic_rounding(x_sub, flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2.0, e[k_sub].float() - (t1 - 1))
        x[k_sub] = x_sub

    # Boundary handling with vectorized operations
    if explim:
        x.masked_fill_((x > xmax) & (x != float('inf')), xmax)
        x.masked_fill_((x < -xmax) & (x != float('-inf')), -xmax)
        min_rep = xmin if subnormal == 0 else xmins
        x.masked_fill_(abs_x < min_rep, 0)

    return x

def _chop_stochastic_rounding_equal(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)

    # Precompute constants
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xboundary = 2 ** emax * (2 - 0.5 * 2 ** (1 - t))

    # Efficient exponent calculation
    abs_x = torch.abs(x)
    _, e = torch.frexp(abs_x)
    e = e - 1
    ktemp = (e < emin) & (e >= emins)

    # Minimize tensor creation
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.empty_like(ktemp, dtype=torch.bool, device=x.device).fill_(False)
        k_norm = torch.empty_like(ktemp, dtype=torch.bool, device=x.device).fill_(True)

    # Normal range: avoid in-place to match original
    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x_norm = x[k_norm] * w
    x_norm = stochastic_rounding_equal(x_norm, flip=flip, p=p, t=t, randfunc=randfunc) * (1 / w)
    x[k_norm] = x_norm

    # Subnormal range: avoid in-place to match original
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.clamp(temp, min=0)  # Optimized with clamp
        w_sub = torch.pow(2.0, t1 - 1 - e[k_sub].float())
        x_sub = x[k_sub] * w_sub
        x_sub = stochastic_rounding_equal(x_sub, flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2.0, e[k_sub].float() - (t1 - 1))
        x[k_sub] = x_sub

    # Boundary handling with vectorized operations
    if explim:
        x.masked_fill_(x >= xboundary, float('inf'))
        x.masked_fill_(x <= -xboundary, float('-inf'))
        min_rep = xmin if subnormal == 0 else xmins
        x.masked_fill_(abs_x < min_rep, 0)

    return x


def round_to_nearest(x, flip=0, p=0.5, t=24, randfunc=None, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
    
    y = torch.abs(x)
    inds = (y - (2 * torch.floor(y / 2))) == 0.5
    y[inds] = y[inds] - 1
    u = torch.round(y)
    u[u == -1] = 0  # Special case
    y = torch.sign(x) * u
    
    if flip:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        temp = torch.randint(0, 2, y.shape, device=x.device)
        k = temp <= p
        if k.any():
            u = torch.abs(y[k])
            b = torch.randint(1, t - 1, u.shape, device=x.device)
            u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
            y[k] = sign(y[k]) * u
    
    return y


def round_towards_plus_inf(x, flip=0, p=0.5, t=24, randfunc=None, **kwargs):
    y = torch.ceil(x)
    
    if flip:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        temp = torch.randint(0, 2, y.shape, device=x.device)
        k = temp <= p
        if k.any():
            u = torch.abs(y[k])
            b = torch.randint(1, t - 1, u.shape, device=x.device)
            u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
            y[k] = sign(y[k]) * u
    
    return y


def round_towards_minus_inf(x, flip=0, p=0.5, t=24, randfunc=None, **kwargs):
    y = torch.floor(x)
    
    if flip:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        temp = torch.randint(0, 2, y.shape, device=x.device)
        k = temp <= p
        if k.any():
            u = torch.abs(y[k])
            b = torch.randint(1, t - 1, u.shape, device=x.device)
            u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
            y[k] = sign(y[k]) * u
    
    return y


def round_towards_zero(x, flip=0, p=0.5, t=24, randfunc=None, **kwargs):
    y = ((x >= 0) | (x == float('-inf'))) * torch.floor(x) + ((x < 0) | (x == float('inf'))) * torch.ceil(x)
    
    if flip:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        temp = torch.randint(0, 2, y.shape, device=x.device)
        k = temp <= p
        if k.any():
            u = torch.abs(y[k])
            b = torch.randint(1, t - 1, u.shape, device=x.device)
            u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
            y[k] = sign(y[k]) * u
    
    return y


def stochastic_rounding(x, flip=0, p=0.5, t=24, randfunc=None):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
    
    y = torch.abs(x)
    frac = y - torch.floor(y)
    
    if not frac.any():
        y = x
    else:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        rnd = randfunc(x.shape)
        j = rnd <= frac
        y[j] = torch.ceil(y[j])
        y[~j] = torch.floor(y[~j])
        y = sign(x) * y
        
        if flip:
            temp = torch.randint(0, 2, y.shape, device=x.device)
            k = temp <= p
            if k.any():
                u = torch.abs(y[k])
                b = torch.randint(1, t - 1, u.shape, device=x.device)
                u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
                y[k] = sign(y[k]) * u
    
    return y

def stochastic_rounding_equal(x, flip=0, p=0.5, t=24, randfunc=None):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
    
    y = torch.abs(x)
    frac = y - torch.floor(y)
    
    if not frac.any():
        y = x
    else:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        rnd = randfunc(x.shape)
        j = rnd <= 0.5
        y[j] = torch.ceil(y[j])
        y[~j] = torch.floor(y[~j])
        y = sign(x) * y
    
    if flip:
        temp = torch.randint(0, 2, y.shape, device=x.device)
        k = temp <= p
        if k.any():
            u = torch.abs(y[k])
            b = torch.randint(1, t - 1, u.shape, device=x.device)
            u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
            y[k] = sign(y[k]) * u
    
    return y


def roundit_test(x, rmode=1, flip=0, p=0.5, t=24, randfunc=None):
    if randfunc is None:
        randfunc = lambda n: torch.randint(0, 2, (n,), device=x.device)
    
    if rmode == 1:
        y = torch.abs(x)
        u = torch.round(y - ((y % 2) == 0.5).float())
        u[u == -1] = 0
        y = torch.sign(x) * u
    elif rmode == 2:
        y = torch.ceil(x)
    elif rmode == 3:
        y = torch.floor(x)
    elif rmode == 4:
        y = ((x >= 0) | (x == float('-inf'))) * torch.floor(x) + ((x < 0) | (x == float('inf'))) * torch.ceil(x)
    elif rmode in (5, 6):
        y = torch.abs(x)
        frac = y - torch.floor(y)
        k = torch.nonzero(frac != 0, as_tuple=True)[0]
        
        if k.numel() == 0:
            y = x
        else:
            rnd = randfunc(k.numel())
            vals = frac[k]
            
            if rmode == 5:
                j = rnd <= vals
            elif rmode == 6:
                j = rnd <= 0.5
                
            y[k[j == 0]] = torch.ceil(y[k[j == 0]])
            y[k[j != 0]] = torch.floor(y[k[j != 0]])
            y = torch.sign(x) * y
    else:
        raise ValueError('Unsupported value of rmode.')
    
    if flip:
        sign = lambda x: torch.sign(x) + (x == 0).float()
        temp = torch.randint(0, 2, y.shape, device=x.device)
        k = temp <= p
        if k.any():
            u = torch.abs(y[k])
            b = torch.randint(1, t - 1, u.shape, device=x.device)
            u = torch.bitwise_xor(u.to(torch.int32), torch.pow(2, b - 1).to(torch.int32)).float()
            y[k] = sign(y[k]) * u
    
    return y

def return_column_order(arr):
    return arr.T.reshape(-1)
