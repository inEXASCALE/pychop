from dataclasses import dataclass

import numpy as np
import gc

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
            
        elif self.prec in {'h','half','fp16','b','bfloat16','s',
                   'single','fp32','d','double','fp64',
                   'q43','fp8-e4m3','q52','fp8-e5m2'}:
            if self.prec in {'q43','fp8-e4m3'}:
                self.t = 4
                self.emax = 7
            elif self.prec in {'q52','fp8-e5m2'}:
                self.t = 3
                self.emax = 15
            elif self.prec in {'h','half','fp16'}:
                self.t = 11
                self.emax = 15
            elif self.prec in {'b','bfloat16'}:
                self.t = 8
                self.emax = 127  
            elif self.prec in {'s','single','fp32'}:
                self.t = 24
                self.emax = 127
            elif self.prec in {'d','double','fp64'}:
                self.t = 53
                self.emax = 1023
        else:
            raise ValueError('Please enter valid prec value.')

        self.u = None
    
            
    def __call__(self, x):
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
                
        y = self.chop_wrapper(x.copy())
        return y
        

    
    def chop_wrapper(self, x):
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip, 
                                explim=self.explim, p=self.p)
    

    @property
    def options(self):
        return options(self.t, 
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