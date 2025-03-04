from dataclasses import dataclass
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

class chop(object):
    def __init__(self, prec='h', subnormal=None, rmode=1, flip=False, explim=1,
                 p=0.5, randfunc=None, customs=None, random_state=0):
        torch.manual_seed(random_state)
        
        self.prec = prec
        self.subnormal = subnormal if subnormal is not None else (prec not in {'b', 'bfloat16'})
        self.rmode = rmode
        self.flip = flip
        self.explim = explim
        self.p = p
        self.randfunc = randfunc or (lambda n: torch.rand(n))

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
        if isinstance(x, (int, str)) and str(x).isnumeric():
            raise ValueError('Chop requires real input values (not int).')
            
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype in (torch.int32, torch.int64):
            x = x.to(torch.float32)
            
        if not x.ndim:
            x = x.unsqueeze(0)
            
        return self.chop_wrapper(x.clone())

    def chop_wrapper(self, x):
        return self._chop(x, t=self.t, emax=self.emax, subnormal=self.subnormal, flip=self.flip, 
                         explim=self.explim, p=self.p, randfunc=self.randfunc)

    @property
    def options(self):
        return options(self.t, self.emax, self.prec, self.subnormal, self.rmode, self.flip, self.explim, self.p)


def _chop_round_to_nearest(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)

    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins

    # PyTorch doesn't have frexp, so we implement it using log2 and floor
    abs_x = torch.abs(x)
    e = torch.floor(torch.log2(abs_x)).int()  # Exponent
    ktemp = (e < emin) & (e >= emins)

    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool)

    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x[k_norm] = round_to_nearest(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w

    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp))
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

    e = torch.floor(torch.log2(torch.abs(x))).int()
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool)

    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x[k_norm] = round_towards_plus_inf(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp))
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
    
    e = torch.floor(torch.log2(torch.abs(x))).int()
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool)

    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x[k_norm] = round_towards_minus_inf(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp))
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
    
    e = torch.floor(torch.log2(torch.abs(x))).int()
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool)

    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x[k_norm] = round_towards_zero(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp))
        x[k_sub] = round_towards_zero(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                      flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        x[(x > xmax) & (x != float('inf'))] = xmax
        x[(x < -xmax) & (x != float('-inf'))] = -xmax
        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        x[k_small] = 0
                
    return x

def _chop_stochastic_rounding(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
        
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    xmax = 2 ** emax * (2 - 2 ** (1 - t))
    
    e = torch.floor(torch.log2(torch.abs(x))).int()
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool)

    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x[k_norm] = stochastic_rounding(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp))
        x[k_sub] = stochastic_rounding(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                       flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        x[(x > xmax) & (x != float('inf'))] = xmax
        x[(x < -xmax) & (x != float('-inf'))] = -xmax
        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        x[k_small] = 0
                
    return x

def _chop_stochastic_rounding_equal(x, t, emax, subnormal=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
    if randfunc is None:
        randfunc = lambda n: torch.rand(n, device=x.device)
        
    emin = 1 - emax
    xmin = 2 ** emin
    emins = emin + 1 - t
    xmins = 2 ** emins
    
    e = torch.floor(torch.log2(torch.abs(x))).int()
    ktemp = (e < emin) & (e >= emins)
              
    if explim:
        k_sub = ktemp
        k_norm = ~ktemp
    else:
        k_sub = torch.zeros_like(ktemp, dtype=torch.bool)
        k_norm = torch.ones_like(ktemp, dtype=torch.bool)

    w = torch.pow(2.0, t - 1 - e[k_norm].float())
    x[k_norm] = stochastic_rounding_equal(x=x[k_norm] * w, flip=flip, p=p, t=t, randfunc=randfunc)
    x[k_norm] *= 1 / w
    
    if k_sub.any():
        temp = emin - e[k_sub]
        t1 = t - torch.max(temp, torch.zeros_like(temp))
        x[k_sub] = stochastic_rounding_equal(x=x[k_sub] * torch.pow(2, t1 - 1 - e[k_sub].float()), 
                                             flip=flip, p=p, t=t, randfunc=randfunc) * torch.pow(2, e[k_sub].float() - (t1 - 1))
        
    if explim:
        xboundary = 2 ** emax * (2 - 0.5 * 2 ** (1 - t))
        x[x >= xboundary] = float('inf')
        x[x <= -xboundary] = float('-inf')
        min_rep = xmin if subnormal == 0 else xmins
        k_small = torch.abs(x) < min_rep
        x[k_small] = 0

    return x

def round_to_nearest(x, flip=0, p=0.5, t=24, randfunc=None, **kwargs):
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