import numpy as np



def return_column_order(A):
    return A.T.reshape(-1)



def roundit(x, rmode=1, flip=0, p=0.5, t=24):
    
    sign = lambda x: np.sign(x) + (x==0)
    
    if hasattr(x, "__len__"):
        is_arr = True
    else:
        is_arr = False
    
    match rmode:
        case 1:
            y = np.abs(x);
            u = np.round(y - ((y % 2) == 0.5))
            
            if is_arr:
                u[np.nonzero(u == -1)] = 0 # Special case, negative argument to ROUND.
                
            y = np.sign(x) * u
            
        case 2:
            y = np.ceil(x)
            
        case 3:
            y = np.floor(x)
            
        case 4:
            y = ((x >= 0) | (x == -np.inf)) * np.floor(x) + ((x < 0) | (x == np.inf)) * np.ceil(x)
            
        case 5 | 6:
            y = np.abs(x); 
            frac = y - np.floor(y);
            k = np.nonzero(frac != 0);
            
            if isempty(k):
                y = x; 
            else:   
                # Uniformly distributed random numbers
                
                rnd = np.random.uniform(0, 1, len(k))
                vals = frac(k)
                
                if len(vals.shape) == 2:
                    vals = return_column_order(vals)
                else:
                    pass
                
                if rmode == 5: # Round up or down with probability prop. to distance.
                    j = rnd <= vals
                else: # Round up or down with equal probability.       
                    j = rnd <= 0.5
                
                y[k(j)] = np.ceil(y[k[j]])
                y[k(j!=0)] = np.floor(y[k[j!=0]])
                y = sign(x)*y
                
        case _:
            print('Unsupported value of options.round.')
            
    if flip:
        temp = np.random.uniform(low=0, high=1, size=y.shape)
        k = np.nonzero(temp <= p); # Indices of elements to have a bit flipped.
        if len(k)==0:
            u = np.abs(y(k));
            # Random bit flip in significand.
            # b defines which bit (1 to p-1) to flip in each element of y.
            # Using SIZE avoids unwanted implicit expansion.
            # The custom (base 2) format is defined by options.params, which is a
            # 2-vector [t,emax] where t is the number of bits in the significand
            # (including the hidden bit) and emax is the maximum value of the
            # exponent.  

            b = randi(t-1, u.shape[0], u.shape[1]);
            b = np.random.uniform(low=1, high=t-1, size=u.shape) # % t is an integer with modulus on [0,15].
            # Flip selected bits.
            u = np.bitwise_xor(u, np.power(2, b-1));
            y[k] = sign(y(k))*u; 
    
    return y
    
    
