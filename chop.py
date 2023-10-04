import numpy as np
    
    
def _chop(x, fmt='single', subnormal=0, rmode=1, flip=0, explim=1, p=0.5, randfunc=None, *argv, **kwargs):
         
        
        if str(x).isnumeric():
            error('Chop requires real input values.')

        if hasattr(x, "__len__"):
            is_arr = True
        else:
            is_arr = False
    
        if not is_arr:
            x = np.array(x, ndmin=1)
            
            
        if randfunc is None:
            randfunc = lambda n: np.random.uniform(0, 1, n)
            
            
        t = None
        emax = None
        
        if fmt in {'h','half','fp16','b','bfloat16','s',
                   'single','fp32','d','double','fp64',
                   'q43','fp8-e4m3','q52','fp8-e5m2'}:
            
            if fmt in {'q43','fp8-e4m3'}:
                t = 4
                emax = 7
            elif fmt in {'q52','fp8-e5m2'}:
                t = 3
                emax = 15
            elif fmt in {'h','half','fp16'}:
                t = 11
                emax = 15
            elif fmt in {'b','bfloat16'}:
                t = 8
                emax = 127  
            elif fmt in {'s','single','fp32'}:
                t = 24
                emax = 127
            elif fmt in {'d','double','fp64'}:
                t = 53
                emax = 1023
            
            
        elif fmt in {'c','custom'}:
            t = customs.t
            emax = customs.emax
            
            if rmode == 1:
                maxfraction = isinstance(x, np.single) * 11 + isinstance(x, np.double) * 25
            else:
                maxfraction = isinstance(x, np.single) * 23 + isinstance(x, np.double) * 52
                
            if t > maxfraction:
                print('Precision of the custom format must be at most')
                
        emin = 1 - emax            # Exponent of smallest normalized number.
        xmin = 2**emin            # Smallest positive normalized number.
        emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
        xmins = pow(2, emins)          # Smallest positive subnormal number.
        xmax = pow(2,emax) * (2-2**(1-t))
        
        
        c = x
        e =  np.floor(np.log2(np.abs(x)) / np.log(2)) - 1
        ktemp = (e < emin) & (e >= emins)
        print("ktemp:", ktemp)
        if explim:
            k_sub = np.nonzero(ktemp)[0]
            k_norm = np.nonzero(ktemp!=1)[0]
        else:
            k_sub = np.array([])
            k_norm = np.arange(1, len(return_column_order(ktemp)) + 1)
        
        print("k_sub:", k_sub)
        print("k_norm:", k_norm)
        
        temp = x[k_norm] * np.power(2, t-1-e[k_norm])
        print("temp:", temp)
        c[k_norm] = roundit(temp, rmode=rmode, t=t) * np.power(2, e[k_norm]-(t-1))
        
        
        print("c[k_norm]:", c[k_norm])
        if k_sub.size != 0:
            temp = emin-e(k_sub)
            t1 = t - np.fmax(temp, np.zeros(temp.shape))
            c[k_sub] = roundit(
                x[k_sub] * np.power(2, t1-1-e[k_sub]), 
                rmode=rmode, 
                randfunc=randfunc,
                t=t
            ) * np.power(2, e[k_sub]-(t1-1));

        
        if explim:
            match rmode:
                case 1 | 6:
                    xboundary = 2**emax * (2-(1/2) * 2**(1-t))
                    c[np.nonzero(x >= xboundary)] = np.inf    # Overflow to +inf.
                    c[np.nonzero(x <= -xboundary)] = -np.inf  # Overflow to -inf.
                    
                case 2:
                    c[np.nonzero(x > xmax)] = np.inf
                    c[np.nonzero((x < -xmax) & (x != -np.inf))] = -xmax
                
                case 3:
                    c[np.nonzero((x > xmax) & (x != np.inf))] = xmax
                    c[np.nonzero(x < -xmax)] = -np.inf
                    
                case 4|5:
                    c[np.nonzero((x > xmax) & (x != np.inf))] = xmax
                    c[np.nonzero((x < -xmax) & (x != -np.inf))] = -xmax
                    
                    
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
    
    
    
