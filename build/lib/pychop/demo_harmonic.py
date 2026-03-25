import numpy as np
from . import customs
from .chop import chop
import copy


np.random.seed(1)


def main():
    print('---------------------------------------------')
    print('Format  Round mode     Sum      No. terms')
    print('---------------------------------------------')
    format_prec = None

    for p in np.arange(0, 5):
        if p == 0:
            prec = 'custom' 
            t = 5
            emax = 3
            params = customs(t, emax)
            format_prec = prec + '  '
            
        elif p == 1:
            prec = 'bfloat16'
            format_prec = prec + ''
            
        elif p == 2:
            prec = 'fp16'
            format_prec = prec + '    '
            
        elif p == 3:
            prec = 'fp8-e4m3'
            format_prec = prec + ''
            
        elif p == 4:
            prec = 'fp8-e5m2'
            format_prec = prec + ''


        for i in np.arange(1, 7):
            rmode = i
            cp = chop(prec=prec, rmode=rmode, customs=params, random_state=20)

            s = 0
            n = 1

            while 1:
                sold = copy.deepcopy(s)
                s = cp(s + cp(1/n))
                if s == sold:
                    break
                n = n + 1
            
            print('{0}    {1:1.0f}      {2:9.4e}      {3:g}'.format(format_prec, i, s[0], n))
        print('------------------------------------------')
