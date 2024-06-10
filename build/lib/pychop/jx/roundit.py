import jax.numpy as jnp
from jax import random
                      
def round_to_nearest(key, x, flip=0, p=0.5, t=24, **kwargs):
    y = jnp.abs(x)
    inds = (y - (2 * jnp.floor(y / 2))) == 0.5
    
    y = y.at[inds].set(y[inds] - 1)
    u = jnp.round(y)
    inds = u == -1
    u = u.at[inds].set(0) # Special case, negative argument to ROUND.
    y = jnp.sign(x) * u
    
    if flip:
        sign = lambda x: jnp.sign(x) + (x==0)
        temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
        k = temp <= p # Indices of elements to have a bit flipped.
        if jnp.any(k):
            u = jnp.abs(y[k])
            b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
            # Flip selected bits.
            u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
            y = y.at[k].set(sign(y[k])*u)
    return y



def round_towards_plus_inf(key, x, flip=0, p=0.5, t=24, **kwargs):
    y = jnp.ceil(x)
            
    if flip:
        sign = lambda x: jnp.sign(x) + (x==0)
        temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
        k = temp <= p # Indices of elements to have a bit flipped.
        if jnp.any(k):
            u = jnp.abs(y[k])
            b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
            # Flip selected bits.
            u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
            y = y.at[k].set(sign(y[k])*u)
    
    return y



def round_towards_minus_inf(key, x, flip=0, p=0.5, t=24, **kwargs):
    y = jnp.floor(x)
            
    if flip:
        sign = lambda x: jnp.sign(x) + (x==0)
        temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
        k = temp <= p # Indices of elements to have a bit flipped.
        if jnp.any(k):
            u = jnp.abs(y[k])
            b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
            # Flip selected bits.
            u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
            y = y.at[k].set(sign(y[k])*u)
    
    return y


def round_towards_zero(key, x, flip=0, p=0.5, t=24, **kwargs):
    y = ((x >= 0) | (x == -jnp.inf)) * jnp.floor(x) + ((x < 0) | (x == jnp.inf)) * jnp.ceil(x)
            
    if flip:
        sign = lambda x: jnp.sign(x) + (x==0)
        temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
        k = temp <= p # Indices of elements to have a bit flipped.
        if jnp.any(k):
            u = jnp.abs(y[k])
            b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
            # Flip selected bits.
            u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
            y = y.at[k].set(sign(y[k])*u)
    
    return y



def stochastic_rounding(key, x, flip=0, p=0.5, t=24, randfunc=None):
    y = jnp.abs(x)
    frac = y - jnp.floor(y)
 
    if jnp.count_nonzero(frac) == 0:
        y = x 
    else:   
        sign = lambda x: jnp.sign(x) + (x==0)
        rnd = randfunc(frac.shape)
        j = rnd <= frac

        y = y.at[j].set(jnp.ceil(y[j])) 
        y = y.at[~j].set(jnp.floor(y[~j])) 
        y = sign(x)*y
                
        if flip:
            temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
            k = temp <= p # Indices of elements to have a bit flipped.
            if jnp.any(k):
                u = jnp.abs(y[k])
                b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
                # Flip selected bits.
                u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
                y = y.at[k].set(sign(y[k])*u)
        
    return y



def stochastic_rounding_equal(key, x, flip=0, p=0.5, t=24, randfunc=None):
    y = jnp.abs(x)
    frac = y - jnp.floor(y)
    
    if jnp.count_nonzero(frac) == 0:
        y = x 
    else:   
        # Uniformly distributed random numbers
        rnd = randfunc(frac.shape)
        j = rnd <= 0.5
        y = y.at[j].set(jnp.ceil(y[j])) 
        y = y.at[~j].set(jnp.floor(y[~j])) 
        y = sign(x)*y
            
    if flip:
        sign = lambda x: jnp.sign(x) + (x==0)
        temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
        k = temp <= p # Indices of elements to have a bit flipped.
        if jnp.any(k):
            u = jnp.abs(y[k])
            b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
            # Flip selected bits.
            u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
            y = y.at[k].set(sign(y[k])*u)
    
    return y



def roundit_test(key, x, rmode=1, flip=0, p=0.5, t=24, randfunc=None):
    if randfunc is None:
        randfunc = lambda n: random.randint(key, shape=n, minval=0, maxval=1)
            
    match rmode:
        case 1:
            y = jnp.abs(x)
            u = jnp.round(y - ((y % 2) == 0.5))
            
            u = u.at[u == -1].set(0) # Special case, negative argument to ROUND.
                
            y = jnp.sign(x) * u
            
        case 2:
            y = jnp.ceil(x)
            
        case 3:
            y = jnp.floor(x)
            
        case 4:
            y = ((x >= 0) | (x == -jnp.inf)) * jnp.floor(x) + ((x < 0) | (x == jnp.inf)) * jnp.ceil(x)
            
        case 5 | 6:
            y = jnp.abs(x)
            frac = y - jnp.floor(y)
            k = jnp.nonzero(frac != 0)[0]
            
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
                    
                y = y.at[k[j==0]].set(jnp.ceil(y[k[j==0]]))
                y = y.set[k[j!=0]].set(jnp.floor(y[k[j!=0]]))
                y = sign(x)*y
                
        case _:
            raise ValueError('Unsupported value of rmode.')
            
    if flip:
        sign = lambda x: jnp.sign(x) + (x==0)
        temp = random.randint(key, shape=y.shape, minval=0, maxval=1)
        k = temp <= p # Indices of elements to have a bit flipped.
        if jnp.any(k):
            u = jnp.abs(y[k])
            
            # Random bit flip in significand.
            # b defines which bit (1 to p-1) to flip in each element of y.
            # Using SIZE avoids unwanted implicit expansion.
            # The custom (base 2) format is defined by options.params, which is a
            # 2-vector [t,emax] where t is the number of bits in the significand
            # (including the hidden bit) and emax is the maximum value of the
            # exponent.  

            
            b = random.randint(key, shape=u.shape, minval=1, maxval=t-1) 
            # Flip selected bits.
            u = jnp.bitwise_xor(jnp.int32(u), jnp.power(2, b-1))
            y = y.at[k].set(sign(y[k])*u)
    
    return y
    
    
    
   
    
    
    
def return_column_order(arr):
    return arr.T.reshape(-1)
