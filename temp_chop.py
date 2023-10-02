    
class chop:
    
    def __init__(x, fmt='single', subnormal=0, rmode=1, flip=0, explim=1, randfunc=False, p=0.5, *argv, **kwargs):
         
        
        self.x = x
        
        # if len(argv) >= 0:
        #    print("Please enter at least two values for argments")
            
        if str(x).isnumeric():
            error('Chop requires real input values.')
        
        
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
                
            if (t > maxfraction)
                print('Precision of the custom format must be at most')
                
        emin = 1 - emax            # Exponent of smallest normalized number.
        xmin = pow(2, emin)            # Smallest positive normalized number.
        emins = emin + 1 - t     # Exponent of smallest positive subnormal number.
        xmins = pow(2, emins)          # Smallest positive subnormal number.
        xmax = pow(2,emax) * (2-2**(1-t))
        
        
        c = x
        e = np.log2(np.abs(x)) - 1
        ktemp = (e < emin) & (e >= emins)
        
        if explim:
            k_sub = np.nonzero(ktemp)
            k_norm = np.nonzero(~ktemp)
        else:
            k_sub = []
            k_norm = 1:len(ktemp) # Do not limit exponent.
    
        
        temp = 
        c[k_norm] = pow2(roundit(pow2(x(k_norm), t-1-e(k_norm)), fpopts), e(k_norm)-(t-1))
        
        return c
