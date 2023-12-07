from dataclasses import dataclass
from roundit import (round_to_nearest, 
                    round_towards_plus_inf, 
                    round_towards_minus_inf, 
                    round_towards_zero, 
                    stochastic_rounding, 
                    stochastic_rounding_equal)
import torch
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

        

        
class TensorChop(object):
    """
    Parameters
    ----------
    prec : str, default='s':
        The target arithmetic format.
    
    subnormal : boolean
        Whether or not support subnormal numbers are supported.
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

    random_state : int, default=0
        Random seed set for stochastic rounding settings.

        
    Methods
    ----------
    chop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """
    def __init__(self, prec='s', subnormal=None, rmode=1, flip=0, explim=1, device='cpu', 
                 p=0.5, randfunc=None, customs=None, random_state=0):
        
        torch.manual_seed(random_state)
        
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
        self.device = device

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
                self.maxfraction = isinstance(x[0], torch.float32) * 11 + isinstance(x[0], torch.float64) * 25
            else:
                self.maxfraction = isinstance(x[0], torch.float32) * 23 + isinstance(x[0], torch.float64) * 52
                
            if self.t > self.maxfraction:
                raise ValueError('Precision of the custom format must be at most')

        else:
            raise ValueError('Please enter valid prec value.')

    
            
    def chop(self, x):
    
        if str(x).isnumeric():
            raise ValueError('Chop requires real input values.')

        if self.randfunc is None:
            self.randfunc = lambda size: torch.rand(size).to(self.device)

        return self._chop(x.to(self.device), 
                     t=self.t, emax=self.emax,
                     subnormal=self.subnormal,
                     flip=self.flip, 
                     explim=self.explim, 
                     p=self.p, 
                     randfunc=self.randfunc,
                     device=self.device
                    )


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
          explim=1, p=0.5, randfunc=None, device='cpu', *argv, **kwargs):
              
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.

    c = torch.clone(x)
    _, e = torch.frexp(torch.abs(x)) 
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.tensor([]).to(device)
        k_norm = torch.arange(0, sum(ktemp.shape)).to(device)

    w = torch.pow(2.0, t-1-e[k_norm])
    c[k_norm] = round_to_nearest(
        x[k_norm] * w, randfunc=randfunc, t=t, flip=flip, p=p, device=device
    ) 

    c[k_norm] *= 1 / w

    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - torch.fmax(temp, torch.zeros(temp.shape).to(device))
        
        c[k_sub] = round_to_nearest(
            x[k_sub] * torch.pow(2, t1-1-e[k_sub]), 
            randfunc=randfunc,
            t=t, flip=flip, p=p, device=device
        ) * torch.pow(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        xboundary = 2**emax * (2- 0.5 * 2**(1-t))
        c[x >= xboundary] = torch.inf    # Overflow to +inf.
        c[x <= -xboundary] = -torch.inf  # Overflow to -inf.
                
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = torch.abs(c) < min_rep

        if subnormal == 0:
            k_round = k_small & (torch.abs(c) >= min_rep/2)
        else:
            k_round = k_small & (torch.abs(c) > min_rep/2)
        
        c[k_round] = torch.sign(c[k_round]) * min_rep
        c[k_small & (k_round != 1)] = 0
                
    return c
    
    

def _chop_round_towards_plus_inf(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, device='cpu', *argv, **kwargs):
              
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    c = torch.clone(x)
    _, e = torch.frexp(torch.abs(x)) 
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.tensor([]).to(device)
        k_norm = torch.arange(0, sum(ktemp.shape)).to(device)

    w = torch.pow(2.0, t-1-e[k_norm])
    c[k_norm] = round_towards_plus_inf(
        x[k_norm] * w, randfunc=randfunc, t=t, flip=flip, p=p, device=device
    ) 

    c[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - torch.fmax(temp, torch.zeros(temp.shape).to(device))
        
        c[k_sub] = round_towards_plus_inf(
            x[k_sub] * torch.pow(2, t1-1-e[k_sub]), 
            randfunc=randfunc,
            t=t, flip=flip, p=p, device=device
        ) * torch.pow(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        c[x > xmax] = torch.inf
        c[(x < -xmax) & (x != -torch.inf)] = -xmax
                
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = torch.abs(c) < min_rep
        
        k_round = k_small & (c > 0) & (c < min_rep)
        c[k_round] = min_rep
        c[k_small & (k_round != 0)] = 0

    return c


def _chop_round_towards_minus_inf(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, device='cpu', *argv, **kwargs):
              
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    c = torch.clone(x)
    _, e = torch.frexp(torch.abs(x)) 
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.tensor([]).to(device)
        k_norm = torch.arange(0, sum(ktemp.shape)).to(device)

    w = torch.pow(2.0, t-1-e[k_norm])
    c[k_norm] = round_towards_minus_inf(
        x[k_norm] * w, randfunc=randfunc, t=t, flip=flip, p=p, device=device
    ) 

    c[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - torch.fmax(temp, torch.zeros(temp.shape).to(device))
        
        c[k_sub] = round_towards_minus_inf(
            x[k_sub] * torch.pow(2, t1-1-e[k_sub]), 
            randfunc=randfunc,
            t=t, flip=flip, p=p, device=device
        ) * torch.pow(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        c[(x > xmax) & (x != torch.inf)] = xmax
        c[x < -xmax] = -torch.inf
        
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = torch.abs(c) < min_rep
        
        k_round = k_small & (c < 0) & (c > -min_rep)
        c[k_round] = -min_rep
        c[k_small & (k_round != 0)] = 0
                
    return c


def _chop_round_towards_zero(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, device='cpu', *argv, **kwargs):
              
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    c = torch.clone(x)
    _, e = torch.frexp(torch.abs(x)) 
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.tensor([]).to(device)
        k_norm = torch.arange(0, sum(ktemp.shape)).to(device)

    w = torch.pow(2.0, t-1-e[k_norm])
    c[k_norm] = round_towards_zero(
        x[k_norm] * w, randfunc=randfunc, t=t, flip=flip, p=p, device=device
    ) 

    c[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - torch.fmax(temp, torch.zeros(temp.shape).to(device))
        
        c[k_sub] = round_towards_zero(
            x[k_sub] * torch.pow(2, t1-1-e[k_sub]), 
            randfunc=randfunc,
            t=t, flip=flip, p=p, device=device
        ) * torch.pow(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        c[(x > xmax) & (x != torch.inf)] = xmax
        c[(x < -xmax) & (x != -torch.inf)] = -xmax
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = torch.abs(c) < min_rep
        c[k_small] = 0
                
    return c


def _chop_stochastic_rounding(x, t, emax, subnormal=1, flip=0, 
          explim=1, p=0.5, randfunc=None, device='cpu', *argv, **kwargs):
              
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.
    xmax = pow(2,emax) * (2-2**(1-t))
    
    c = torch.clone(x)
    _, e = torch.frexp(torch.abs(x)) 
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.tensor([]).to(device)
        k_norm = torch.arange(0, sum(ktemp.shape)).to(device)

    w = torch.pow(2.0, t-1-e[k_norm])
    c[k_norm] = stochastic_rounding(
        x[k_norm] * w, randfunc=randfunc, t=t, flip=flip, p=p, device=device
    ) 

    c[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - torch.fmax(temp, torch.zeros(temp.shape).to(device))
        
        c[k_sub] = stochastic_rounding(
            x[k_sub] * torch.pow(2, t1-1-e[k_sub]), 
            randfunc=randfunc,
            t=t, flip=flip, p=p, device=device
        ) * torch.pow(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        c[(x > xmax) & (x != torch.inf)] = xmax
        c[(x < -xmax) & (x != -torch.inf)] = -xmax
          
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = torch.abs(c) < min_rep
        c[k_small] = 0
                
    return c



def _chop_stochastic_rounding_equal(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, 
                                    randfunc=None, device='cpu', *argv, **kwargs):
              
    emin = 1 - emax            # Exponent of smallest normalized number.
    xmin = 2**emin            # Smallest positive normalized number.
    emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
    xmins = pow(2, emins)          # Smallest positive subnormal number.

    c = torch.clone(x)
    _, e = torch.frexp(torch.abs(x)) 
    e = e - 1
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.tensor([]).to(device)
        k_norm = torch.arange(0, sum(ktemp.shape)).to(device)

    w = torch.pow(2.0, t-1-e[k_norm])
    c[k_norm] = stochastic_rounding_equal(
        x[k_norm] * w, randfunc=randfunc, t=t, flip=flip, p=p, device=device
    ) 

    c[k_norm] *= 1 / w
    
    if k_sub.size != 0:
        temp = emin-e[k_sub]
        t1 = t - torch.fmax(temp, torch.zeros(temp.shape).to(device))
        
        c[k_sub] = stochastic_rounding_equal(
            x[k_sub] * torch.pow(2, t1-1-e[k_sub]), 
            randfunc=randfunc,
            t=t, flip=flip, p=p, device=device
        ) * torch.pow(2, e[k_sub]-(t1-1))
        del temp, t1
        
    del w; gc.collect()
        
    if explim:
        xboundary = 2**emax * (2- 0.5 * 2**(1-t))
        c[x >= xboundary] = torch.inf    # Overflow to +inf.
        c[x <= -xboundary] = -torch.inf  # Overflow to -inf.
        
        # Round to smallest representable number or flush to zero.
        if subnormal == 0:
            min_rep = xmin
        else:
            min_rep = xmins

        k_small = torch.abs(c) < min_rep
        c[k_small] = 0
                
    return c
