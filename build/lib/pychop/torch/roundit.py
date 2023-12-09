import torch
                      
def round_to_nearest(x, flip=0, p=0.5, t=24, device='cpu', **kwargs):
    sign = lambda x: torch.sign(x) + (x==0)
    
    y = torch.abs(x)
    u = torch.round(y - ((y % 2) == 0.5).to(torch.long))
    u[u == -1] = 0 # Special case, negative argument to ROUND.
    y = torch.sign(x) * u
            
    if flip:
        temp = torch.rand(size=y.shape).to(device)
        k = temp <= p; # Indices of elements to have a bit flipped.
        if torch.any(k):
            u = torch.abs(y[k])
            b = torch.randint(1, t-1, size=u.shape).to(device)
            # Flip selected bits.
            u = torch.bitwise_xor(u.to(torch.long), torch.pow(2, b-1))
            
            y[k] = sign(y[k])*u
    
    return y



def round_towards_plus_inf(x, flip=0, p=0.5, t=24, device='cpu', **kwargs):
    sign = lambda x: torch.sign(x) + (x==0)
    
    y = torch.ceil(x)
            
    if flip:
        temp = torch.rand(size=y.shape).to(device)
        k = temp <= p; # Indices of elements to have a bit flipped.
        if torch.any(k):
            u = torch.abs(y[k])
            b = torch.randint(1, t-1, size=u.shape).to(device)
            # Flip selected bits.
            u = torch.bitwise_xor(u.to(torch.long), torch.pow(2, b-1))
            y[k] = sign(y[k])*u
    
    return y



def round_towards_minus_inf(x, flip=0, p=0.5, t=24, device='cpu', **kwargs):

    sign = lambda x: torch.sign(x) + (x==0)
    y = torch.floor(x)
            
    if flip:
        temp = torch.rand(size=y.shape).to(device)
        k = temp <= p; # Indices of elements to have a bit flipped.
        if torch.any(k):
            u = torch.abs(y[k])
            
            b = torch.randint(1, t-1, size=u.shape).to(device)
            # Flip selected bits.
            u = torch.bitwise_xor(u.to(torch.long), torch.pow(2, b-1))
            y[k] = sign(y[k])*u
    
    return y


def round_towards_zero(x, flip=0, p=0.5, t=24, device='cpu', **kwargs):
    sign = lambda x: torch.sign(x) + (x==0)
    
    y = ((x >= 0) | (x == -torch.inf)) * torch.floor(x) + ((x < 0) | (x == torch.inf)) * torch.ceil(x)
            
    if flip:
        temp = torch.rand(size=y.shape).to(device)
        k = temp <= p; # Indices of elements to have a bit flipped.
        if torch.any(k):
            u = torch.abs(y[k])
            
            b = torch.randint(1, t-1, size=u.shape).to(device)
            # Flip selected bits.
            u = torch.bitwise_xor(u.to(torch.long), torch.pow(2, b-1))
            y[k] = sign(y[k])*u
    
    return y



def stochastic_rounding(x, flip=0, p=0.5, t=24, randfunc=None, device='cpu'):
    sign = lambda x: torch.sign(x) + (x==0)
    
    if randfunc is None:
        randfunc = lambda n: torch.rand(n)
            
    y = torch.abs(x)
    frac = y - torch.floor(y)
    
    if torch.count_nonzero(frac) == 0:
        y = x; 
    else:   
        rnd = randfunc(frac.shape)
        j = rnd <= frac
            
        y[j] = torch.ceil(y[j])
        y[~j] = torch.floor(y[~j])
        y = sign(x)*y
                
        if flip:
            temp = torch.rand(size=y.shape).to(device)
            k = temp <= p; # Indices of elements to have a bit flipped.
            if torch.any(k):
                u = torch.abs(y[k])
                
                b = torch.randint(1, t-1, size=u.shape).to(device)
                # Flip selected bits.
                u = torch.bitwise_xor(u.to(torch.long), torch.pow(2, b-1))
                y[k] = sign(y[k])*u
        
    return y



def stochastic_rounding_equal(x, flip=0, p=0.5, t=24, randfunc=None, device='cpu'):
    sign = lambda x: torch.sign(x) + (x==0)
    
    if randfunc is None:
        randfunc = lambda n: torch.rand(n)
            
    y = torch.abs(x)
    frac = y - torch.floor(y)

    if torch.count_nonzero(frac) == 0:
        y = x; 
    else:   
        # Uniformly distributed random numbers
        rnd = randfunc(frac.shape)
        j = rnd <= 0.5
            
        y[j] = torch.ceil(y[j])
        y[~j] = torch.floor(y[~j])
        y = sign(x)*y
    
    if flip:
        temp = torch.rand(size=y.shape).to(device)
        k = temp <= p; # Indices of elements to have a bit flipped.
        if torch.any(k):
            u = torch.abs(y[k])
            
            b = torch.randint(1, t-1, size=u.shape).to(device)
            # Flip selected bits.
            u = torch.bitwise_xor(u.to(torch.long), torch.pow(2, b-1))
            y[k] = sign(y[k])*u
    
    return y


