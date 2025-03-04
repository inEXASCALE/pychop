from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import random

@dataclass
class customs:
    t: int
    emax: int
        
@dataclass
class options:
    t: int
    emax: int
    prec: int
    subnormal: bool
    rmode: bool
    flip: bool
    explim: bool
    p: float

class chop(object):
    def __init__(self, prec='h', subnormal=None, rmode=1, flip=False, explim=1,
                 p=0.5, randfunc=None, customs=None, random_state=0):
        
        self.key = random.PRNGKey(random_state)
        
        self.prec = prec
        
        if subnormal is not None:
            self.subnormal = subnormal
        else:
            if self.prec in {'b', 'bfloat16'}:
                self.subnormal = False
            else:
                self.subnormal = True
            
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        
        self.randfunc = randfunc
        if self.rmode == 1:
            self._chop = _chop_round_to_nearest
        elif self.rmode == 2:
            self._chop = _chop_round_towards_plus_inf
        elif self.rmode == 3:
            self._chop = _chop_round_towards_minus_inf
        elif self.rmode == 4:
            self._chop = _chop_round_towards_zero
        elif self.rmode == 5:
            self._chop = _chop_stochastic_rounding
        elif self.rmode == 6:
            self._chop = _chop_stochastic_rounding_equal
        else:
            raise ValueError('Unsupported value of rmode.')

        if customs is not None:
            self.t = customs.t
            self.emax = customs.emax
        elif self.prec in {'h', 'half', 'fp16', 'b', 'bfloat16', 's',
                           'single', 'fp32', 'd', 'double', 'fp64',
                           'q43', 'fp8-e4m3', 'q52', 'fp8-e5m2'}:
            if self.prec in {'q43', 'fp8-e4m3'}:
                self.t = 4
                self.emax = 7
            elif self.prec in {'q52', 'fp8-e5m2'}:
                self.t = 3
                self.emax = 15
            elif self.prec in {'h', 'half', 'fp16'}:
                self.t = 11
                self.emax = 15
            elif self.prec in {'b', 'bfloat16'}:
                self.t = 8
                self.emax = 127  
            elif self.prec in {'s', 'single', 'fp32'}:
                self.t = 24
                self.emax = 127
            elif self.prec in {'d', 'double', 'fp64'}:
                self.t = 53
                self.emax = 1023
        else:
            raise ValueError('Please enter valid prec value.')

        self.u = None
    
    def __call__(self, x):
        if isinstance(x, (int, str)) and str(x).isnumeric():
            raise ValueError('Chop requires real input values (not int).')
            
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x, dtype=jnp.float32)
        elif x.dtype in (jnp.int32, jnp.int64):
            x = x.astype(jnp.float32)
            
        if not x.ndim:
            x = jnp.expand_dims(x, 0)
            
        if hasattr(self, 'customs'):
            if self.rmode == 1:
                self.maxfraction = (x.dtype == jnp.float32) * 11 + (x.dtype == jnp.float64) * 25
            else:
                self.maxfraction = (x.dtype == jnp.float32) * 23 + (x.dtype == jnp.float64) * 52
            if self.t > self.maxfraction:
                raise ValueError('Precision of the custom format must be at most')
                
        # Split PRNG key for each call to ensure randomness
        self.key, subkey = random.split(self.key)
        y = self.chop_wrapper(x, subkey)
        return y

    def chop_wrapper(self, x, key):
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip, 
                         explim=self.explim, p=self.p, key=key)

    @property
    def options(self):
        return options(self.t, self.emax, self.prec, self.subnormal, self.rmode, self.flip, self.explim, self.p)

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

def round_to_nearest(x, flip=0, p=0.5, t=24, randfunc=None, key=None, **kwargs):
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

def roundit_test(x, rmode=1, flip=0, p=0.5, t=24, randfunc=None, key=None):
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
    return arr.T.reshape(-1)