from dataclasses import dataclass
from .roundit import rounditcase1, rounditcase2, rounditcase3, rounditcase4, rounditcase5, rounditcase6, rounditcase1rountitTest
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
    random_state: int
    prec: int
    subnormal: bool
    rmode: bool
    flip: bool
    explim: bool
    p: float
    input_prec: str

        

        
class chop(object):
    def __init__(self, prec='single', subnormal=None, rmode=1, flip=0, explim=1, input_prec='double',
                 p=0.5, randfunc=None, customs=None, random_state=0):
        
        if input_prec in {'d', 'double', float, np.double}:
            self.input_prec = np.double
        else:
            self.input_prec = np.float
        
        np.random.seed(random_state)
        
        self.prec = prec
        
        if subnormal is not None:
            self.subnormal = subnormal
        else:
            if self.prec in {'b','bfloat16'}:
                self.subnormal = 0
            else:
                self.subnormal = 1
            
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        self.customs = customs
        self.randfunc = randfunc

        if self.rmode == 1:
            self.roundit = rounditcase1
            
        elif self.rmode == 2:
            self.roundit = rounditcase2
            
        elif self.rmode == 3:
            self.roundit = rounditcase3
            
        elif self.rmode == 4:
            self.roundit = rounditcase4
            
        elif self.rmode == 5:
            self.roundit = rounditcase5
            
        elif self.rmode == 6:
            self.roundit = rounditcase6

        else:
            raise ValueError('Unsupported value of rmode.')

        
        if self.prec in {'h','half','fp16','b','bfloat16','s',
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
            
            
        elif self.prec in {'c','custom'}:
            self.t = customs.t
            self.emax = customs.emax
            
            if self.rmode == 1:
                self.maxfraction = isinstance(x[0], np.single) * 11 + isinstance(x[0], np.double) * 25
            else:
                self.maxfraction = isinstance(x[0], np.single) * 23 + isinstance(x[0], np.double) * 52
                
            if self.t > self.maxfraction:
                raise ValueError('Precision of the custom format must be at most')

        else:
            raise ValueError('Please enter valid prec value.')

    
            
    def chop(self, x):
        return _chop(x, 
                     t=self.t, emax=self.emax,
                     input_prec=self.input_prec,
                     subnormal=self.subnormal,
                     rmode=self.rmode,
                     flip=self.flip, 
                     explim=self.explim, 
                     p=self.p, 
                     randfunc=self.randfunc
                     func_roundit=self.roundit
                    )

    
    
    def options(self):
        return options(self.t, 
                       self.emax,
                       self.random_state,
                       self.prec,
                       self.subnormal,
                       self.rmode,
                       self.flip,
                       self.explim,
                       self.p,
                       self.input_prec,
                      )
    
    


    
def _chop(x, t, emax, input_prec=np.double, subnormal=1, rmode=1, flip=0, 
          explim=1, p=0.5, randfunc=None, func_roundit=rountitcase1, *argv, **kwargs):
              
    if str(x).isnumeric():
        raise ValueError('Chop requires real input values.')
    
    x = input_prec(x)

    if not hasattr(x, "__len__"):
        x = np.array(x, ndmin=1)

    if randfunc is None:
        randfunc = lambda n: np.random.uniform(0, 1, n)
        
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    
    c = x
    _, e = np.frexp(np.abs(x)) 
    e = np.array(e - 1, ndmin=1)
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp.astype(bool)
        k_norm = ~ktemp
    else:
        k_sub = np.array([])
        k_norm = np.arange(0, len(return_column_order(ktemp)))

    w = np.power(2.0, t-1-e[k_norm])
    c[k_norm] = func_roundit(
        x[k_norm] * w, t=t
    ) 

    c[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - np.fmax(temp, np.zeros(temp.shape))
        
        c[k_sub] = roundit(
            x[k_sub] * np.power(2, t1-1-e[k_sub]), 
            rmode=rmode, 
            randfunc=randfunc,
            t=t
        ) * np.power(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        match rmode:
            case 1 | 6:
                xboundary = 2**emax * (2- 0.5 * 2**(1-t))
                c[x >= xboundary] = np.inf    # Overflow to +inf.
                c[x <= -xboundary] = -np.inf  # Overflow to -inf.
                
            case 2:
                c[x > xmax] = np.inf
                c[(x < -xmax) & (x != -np.inf)] = -xmax
            
            case 3:
                c[(x > xmax) & (x != np.inf)] = xmax
                c[x < -xmax] = -np.inf
                
                
            case 4|5:
                c[(x > xmax) & (x != np.inf)] = xmax
                c[(x < -xmax) & (x != -np.inf)] = -xmax
                
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin;
        else:
            min_rep = xmins;

        k_small = np.abs(c) < min_rep;
        
        match rmode:
            case 1:
                if subnormal == 0:
                    k_round = k_small & (np.abs(c) >= min_rep/2)
                else:
                    k_round = k_small & (np.abs(c) > min_rep/2)
              
                c[k_round] = np.sign(c[k_round]) * min_rep
                c[k_small & (k_round != 1)] = 0
                
            case 2:
                k_round = k_small & (c > 0) & (c < min_rep)
                c[k_round] = min_rep
                c[k_small & (k_round != 0)] = 0
                
            case 3:
                k_round = k_small & (c < 0) & (c > -min_rep)
                c[k_round] = -min_rep
                c[k_small & (k_round != 0)] = 0
                
            case 4 | 5 | 6:
                c[k_small] = 0
                
    return c
    
    
    
    
    
    
    
    
    
    
def return_column_order(arr):
    return arr.T.reshape(-1)
    
    
