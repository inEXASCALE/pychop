"""NumPy floating-point chopping backend with deterministic and stochastic modes."""

import numpy as np
from .._fault_formats import FORMAT_PARAMS, NO_SUBNORMAL_DEFAULT, resolve_customs
from functools import wraps
   

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
    Chop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """

    def __init__(self, prec='h', subnormal=None, rmode=1, flip=False, explim=1,
                 p=0.5, randfunc=None, customs=None, random_state=0):
        
        np.random.seed(random_state)
        
        self.prec = prec
        self.ocp_e4m3 = prec == "fp8-e4m3" and customs is None
        
        if subnormal is not None:
            self.subnormal = subnormal
        else:
            if self.prec in NO_SUBNORMAL_DEFAULT:
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
            0: _chop_round_to_nearest_odd,
            1: _chop_round_to_nearest,
            2: _chop_round_towards_plus_inf,
            3: _chop_round_towards_minus_inf,
            4: _chop_round_towards_zero,
            5: _chop_stochastic_rounding,
            6: _chop_stochastic_rounding_equal,
            10: _chop_cadna_rounding
        }.get(rmode, lambda *args: raise_value_error('Unsupported rmode'))

        if rmode == 10:
            from ..cadna_random import CADNARandomGenerator
            self._cadna_gen = CADNARandomGenerator(seed=random_state, backend="numpy")
        else:
            self._cadna_gen = None

        # Set precision parameters
        if customs:
            customs = resolve_customs(customs)
            self.t, self.emax = customs.t, customs.emax
        else:
            prec_map = FORMAT_PARAMS
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
                
        if self.ocp_e4m3:
            return _quantize_fixed_grid(
                x, self.t, self.emax, rmode=self.rmode, subnormal=self.subnormal,
                flip=self.flip, explim=self.explim, p=self.p,
                randfunc=self.randfunc, random_gen=self._cadna_gen, ocp_e4m3=True,
                overflow=getattr(self, "overflow", "saturate"),
            )
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip,
                          explim=self.explim, p=self.p, randfunc=self.randfunc,
                          random_gen=self._cadna_gen)
    

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
    
    
    
def _quantize_fixed_grid(x, t, emax, rmode=1, subnormal=1, flip=0,
                         explim=1, p=0.5, randfunc=None, random_gen=None,
                         ocp_e4m3=False, overflow="saturate"):
    """Round once on the target grid, using finite intermediate scaling."""
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)

    emin = -6 if ocp_e4m3 else 1 - emax
    finite = np.isfinite(x)
    active = finite & (x != 0)
    original_sign = np.signbit(x)
    no_subnormal_tie = (
        active & (np.abs(x) == 2.0 ** (emin - 1))
        if explim and not subnormal and rmode == 1 else None
    )
    if not np.any(active):
        if ocp_e4m3 and explim:
            if overflow == "saturate":
                x[np.isposinf(x)] = 448.0
                x[np.isneginf(x)] = -448.0
            else:
                x[np.isinf(x)] = np.nan
        return x

    _, frexp_exp = np.frexp(np.abs(x))
    exponent = frexp_exp - 1
    if explim:
        # The target subnormal grid has a fixed spacing. With subnormals
        # disabled, only zero and the smallest normal remain below emin.
        grid_exp = np.maximum(exponent, emin)
        if not subnormal:
            grid_exp = np.where(exponent < emin, emin + t - 1, grid_exp)
    else:
        grid_exp = exponent
    step_exp = grid_exp - (t - 1)
    scaled = np.ldexp(x[active], -step_exp[active])

    if rmode == 0:
        rounded = np.rint(scaled)
        halfway = np.abs(scaled - np.trunc(scaled)) == 0.5
        rounded[halfway] = np.copysign(np.floor(np.abs(scaled[halfway])) +
                                           (np.floor(np.abs(scaled[halfway])) % 2 == 0),
                                           scaled[halfway])
    else:
        rounder = {
            1: round_to_nearest, 2: round_towards_plus_inf,
            3: round_towards_minus_inf, 4: round_towards_zero,
            5: stochastic_rounding, 6: stochastic_rounding_equal,
            10: cadna_style_rounding,
        }[rmode]
        round_kwargs = dict(flip=flip, p=p, t=t, randfunc=randfunc)
        if rmode == 10:
            round_kwargs["random_gen"] = random_gen
        rounded = rounder(x=scaled, **round_kwargs)
    x[active] = np.ldexp(rounded, step_exp[active])

    if explim:
        if ocp_e4m3:
            # OCP E4M3 has no infinity and reserves the 480 code for NaN.
            if overflow == "saturate":
                np.clip(x, -448.0, 448.0, out=x)
            else:
                x[np.abs(x) > 448.0] = np.nan
        else:
            xmax = 2.0 ** emax * (2.0 - 2.0 ** (1 - t))
            if rmode in (1, 6):
                boundary = 2.0 ** emax * (2.0 - 0.5 * 2.0 ** (1 - t))
                x[x >= boundary] = np.inf
                x[x <= -boundary] = -np.inf
            elif rmode == 2:
                x[x > xmax] = np.inf
                x[(x < -xmax) & (x != -np.inf)] = -xmax
            elif rmode == 3:
                x[(x > xmax) & (x != np.inf)] = xmax
                x[x < -xmax] = -np.inf
            else:
                x[(x > xmax) & (x != np.inf)] = xmax
                x[(x < -xmax) & (x != -np.inf)] = -xmax

    # Preserve the existing flush-to-zero halfway convention.
    if no_subnormal_tie is not None:
        x[no_subnormal_tie] = np.where(
            original_sign[no_subnormal_tie], -2.0 ** emin, 2.0 ** emin
        )
    zeros = x == 0
    x[zeros] = np.where(original_sign[zeros], -0.0, 0.0)
    return x


def _make_fixed_grid_rounder(rmode):
    def rounder(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5,
                randfunc=None, *args, **kwargs):
        return _quantize_fixed_grid(
            x, t, emax, rmode=rmode, subnormal=subnormal, flip=flip,
            explim=explim, p=p, randfunc=randfunc,
            random_gen=kwargs.get("random_gen"),
            ocp_e4m3=kwargs.get("ocp_e4m3", False),
        )
    return rounder


_chop_round_to_nearest_odd = _make_fixed_grid_rounder(0)
_chop_round_to_nearest = _make_fixed_grid_rounder(1)
_chop_round_towards_plus_inf = _make_fixed_grid_rounder(2)
_chop_round_towards_minus_inf = _make_fixed_grid_rounder(3)
_chop_round_towards_zero = _make_fixed_grid_rounder(4)
_chop_stochastic_rounding = _make_fixed_grid_rounder(5)
_chop_stochastic_rounding_equal = _make_fixed_grid_rounder(6)


def raise_value_error(msg):
    """Raise ``ValueError`` with a backend-compatible helper signature.

    Parameters
    ----------
    msg : str
        Error message to report.
    """
    raise ValueError(msg)







                      
def round_to_nearest(x, flip=0, p=0.5, t=24, **kwargs):
    """Round scaled mantissas to nearest with ties to even.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
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
    """Round scaled mantissas toward positive infinity.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
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
    """Round scaled mantissas toward negative infinity.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
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
    """Round scaled mantissas toward zero.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
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
    """Stochastically round scaled mantissas with distance-proportional odds.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.
    randfunc : callable
        Function returning uniform random values in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
    y = np.abs(x)
    frac = y - np.floor(y)
 
    if np.count_nonzero(frac) == 0:
        y = x 
    else:   
        sign = lambda x: np.sign(x) + (x==0)
        rnd = randfunc(frac.size).reshape(frac.shape)
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
    """Stochastically round scaled mantissas with equal up/down odds.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.
    randfunc : callable
        Function returning uniform random values in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
    y = np.abs(x)
    frac = y - np.floor(y)
    
    if np.count_nonzero(frac) == 0:
        y = x 
    else:   
        # Uniformly distributed random numbers
        sign = lambda x: np.sign(x) + (x==0)
        rnd = randfunc(frac.size).reshape(frac.shape)
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

# ============================================================
# CADNA-style Random Rounding via true sign-bit flip trick
# rmode=10
# ============================================================

import numpy as np


def _random_bits(shape, random_gen=None, randfunc=None):
    """
    Generate random bits, values 0 or 1.

    Priority:
        1. random_gen.random_bits(shape)
        2. randfunc(size=shape)
        3. np.random.randint fallback
    """
    if random_gen is not None:
        return np.asarray(random_gen.random_bits(shape), dtype=np.uint8)

    if randfunc is not None:
        r = randfunc(size=shape)
        return (np.asarray(r) < 0.5).astype(np.uint8)

    return np.random.randint(0, 2, size=shape, dtype=np.uint8)


def _cadna_bit_flip_numpy(x, bits):
    """
    True CADNA-style sign-bit flip for NumPy arrays.

    bits = 0 -> keep x
    bits = 1 -> flip IEEE-754 sign bit

    Supports float32 and float64.
    """
    x = np.asarray(x)

    if x.dtype == np.float64:
        y = np.array(x, copy=True)
        bits = np.broadcast_to(np.asarray(bits, dtype=np.uint8), y.shape)
        y_int = y.view(np.uint64)
        mask = bits.astype(np.uint64) * np.uint64(1 << 63)
        return (y_int ^ mask).view(np.float64)

    if x.dtype == np.float32:
        y = np.array(x, copy=True)
        bits = np.broadcast_to(np.asarray(bits, dtype=np.uint8), y.shape)
        y_int = y.view(np.uint32)
        mask = bits.astype(np.uint32) * np.uint32(1 << 31)
        return (y_int ^ mask).view(np.float32)

    raise ValueError(f"Unsupported dtype for CADNA bit flip: {x.dtype}")


def cadna_style_rounding(x, flip=0, p=0.5, t=24, randfunc=None, random_gen=None):
    """
    CADNA-style stochastic rounding using the true sign-bit flip trick.

    Input x is assumed to already be scaled by 2^(t-1-e), so rounding x to
    an integer corresponds to rounding the significand.

    This implements the CADNA-style directed random rounding idea:

        bit = 0:
            upward rounding:
                y = ceil(x)

        bit = 1:
            downward rounding simulated by sign-bit flip:
                y = bit_flip(ceil(bit_flip(x)))

            Since bit_flip(x) == -x when bit=1:
                y = -ceil(-x) == floor(x)

    Therefore this is equivalent to random choice between upward and downward
    directed rounding, but implemented through sign-bit flipping.

    Parameters
    ----------
    x : np.ndarray
        Input array, already scaled by 2^(t-1-e).
    flip : int
        Optional soft-error bit flip simulation. Not part of CADNA rounding.
    p : float
        Probability for optional soft-error simulation.
    t : int
        Significand bits.
    randfunc : callable
        Optional random function.
    random_gen : CADNARandomGenerator
        CADNA-style random generator. Should be reused across calls.

    Returns
    -------
    np.ndarray
        Rounded array.
    """
    x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    else:
        x = x.copy()

    # Keep original dtype if possible.
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)

    bits = _random_bits(x.shape, random_gen=random_gen, randfunc=randfunc)

    finite = np.isfinite(x)

    # CADNA bit-flip trick:
    #
    # bit=0: x_flipped = x,    ceil(x),     flip back -> ceil(x)
    # bit=1: x_flipped = -x,   ceil(-x),    flip back -> -ceil(-x) = floor(x)
    x_flipped = _cadna_bit_flip_numpy(x, bits)

    y_upward = np.ceil(x_flipped)

    y = _cadna_bit_flip_numpy(y_upward, bits)

    # Preserve NaN and Inf exactly.
    y[~finite] = x[~finite]

    # Optional soft-error simulation.
    # This is NOT part of CADNA stochastic arithmetic itself.
    if flip:
        y = y.copy()
        k = np.isfinite(y) & (y != 0) & (np.random.random(size=y.shape) < p)

        if np.any(k) and t > 2:
            u = np.abs(y[k]).astype(np.int64, copy=False)
            b = np.random.randint(low=1, high=t - 1, size=u.shape)
            mask = np.left_shift(np.int64(1), b - 1)
            u = np.bitwise_xor(u, mask)
            y[k] = (np.sign(y[k]) + (y[k] == 0)) * u

    return y


def _chop_cadna_rounding(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5,
                         randfunc=None, random_gen=None, *argv, **kwargs):
    """CADNA rounding on the target grid without overflowing scale factors."""
    return _quantize_fixed_grid(
        x, t, emax, rmode=10, subnormal=subnormal, flip=flip, explim=explim,
        p=p, randfunc=randfunc, random_gen=random_gen,
    )


def _preserve_signed_zero(rounding_function):
    """Restore the input sign whenever a floating-point result rounds to zero."""
    @wraps(rounding_function)
    def wrapped(x, *args, **kwargs):
        negative = np.signbit(x)
        input_zero = x == 0
        result = rounding_function(x, *args, **kwargs)
        restore = (result == 0) | (input_zero & np.isnan(result))
        result[restore] = np.where(negative[restore], -0.0, 0.0)
        return result
    return wrapped


_chop_round_to_nearest = _preserve_signed_zero(_chop_round_to_nearest)
_chop_round_to_nearest_odd = _preserve_signed_zero(_chop_round_to_nearest_odd)
_chop_round_towards_plus_inf = _preserve_signed_zero(_chop_round_towards_plus_inf)
_chop_round_towards_minus_inf = _preserve_signed_zero(_chop_round_towards_minus_inf)
_chop_round_towards_zero = _preserve_signed_zero(_chop_round_towards_zero)
_chop_stochastic_rounding = _preserve_signed_zero(_chop_stochastic_rounding)
_chop_stochastic_rounding_equal = _preserve_signed_zero(_chop_stochastic_rounding_equal)
_chop_cadna_rounding = _preserve_signed_zero(_chop_cadna_rounding)


def roundit_test(x, rmode=1, flip=0, p=0.5, t=24, randfunc=None):
    """Apply a standalone rounding mode for implementation checks.

    Parameters
    ----------
    x : numpy.ndarray
        Scaled mantissa values.
    rmode : int, default=1
        Rounding mode to apply.
    flip : bool or int, default=0
        Whether to apply the optional soft-error bit flip.
    p : float, default=0.5
        Bit-flip probability when ``flip`` is enabled.
    t : int, default=24
        Significand precision used to select a flippable bit.
    randfunc : callable, default=None
        Optional random-number function used by stochastic modes.

    Returns
    -------
    numpy.ndarray
        Rounded mantissas.
    """
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
    """Return a flattened copy in column-major traversal order.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Flattened array using column-major ordering.
    """
    return arr.T.reshape(-1)
