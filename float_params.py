import numpy as np


def float_params(*argv):
    if len(argv) < 1:
        precs = 'bhtsdq'
        
        print('-------------------------------------------------------------------------------')
        print('       ' + \
              ' | '.join(['   u   ', '  xmins  ', 
                          ' xmin ', '   xmax  ', 
                          ' p ', ' emins ', 
                          ' emin ', 'emax'
                         ]
                        ) + '')
        print('-------------------------------------------------------------------------------')
        for j in np.arange(-2, len(precs)):
            
            match j:
                case -2:
                    prec = 'q43'
                    
                case -1:
                    prec = 'q52'
            
                case _:
                    prec = precs[j]
            
            (u,xmins,xmin,xmax,p,emins,emin,emax) = float_params(prec)
            
            if prec in ['q43', 'q52']:
                print(
                    f'{prec:s}  {u:9.2e}  {xmins:9.2e}  {xmin:9.2e}  {xmax:9.2e}  '
                    f'{p:3.0f}  {emins:7.0f}  {emin:7.0f}  {emax:6.0f}'
                )
            else:
                print(
                    f' {prec:s}   {u:9.2e}  {xmins:9.2e}  {xmin:9.2e}  {xmax:9.2e}  '
                    f'{p:3.0f}  {emins:7.0f}  {emin:7.0f}  {emax:6.0f}'
                )
        print('-------------------------------------------------------------------------------')
        
    else:
        match argv[0]:
            case 'q43' | 'fp8-e4m3':
                p = 4
                emax = 7
            case 'q52' | 'fp8-e5m2':
                p = 3
                emax = 15
            case 'b' | 'bfloat16':
                p = 8
                emax = 127  
            case 'h' | 'half' | 'fp16':
                p = 11
                emax = 15
            case 't' | 'tf32':
                p = 11
                emax = 127 
            case 's' | 'single' | 'fp32':
                p = 24
                emax = 127
            case 'd' | 'double' | 'fp64':
                p = 53
                emax = 1023
            case 'q' | 'quadruple' | 'fp128':
                p = 113
                emax = 16383

            case _:
                raise ValueError('Please specify a parameter supported by the software.')
                
        emin = 1-emax
        emins = emin + 1 - p   
        xmins = 2**emins
        xmin = 2**emin
        
        try:
            xmax = 2**emax * (2-2**(1-p))
        except OverflowError:
            xmax = float('inf')
        
        u = 2**(-p)
        
        return u,xmins,xmin,xmax,p,emins,emin,emax
