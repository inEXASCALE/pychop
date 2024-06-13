# This API follows https://github.com/higham/chop/blob/master/float_params.m

import numpy as np
import pandas as pd


def binary_mark(value):
    """Covert value into exponential form of 2"""
    try:
        exp = int(np.round(np.log2(value)))
        return '2^'+str(exp) 
    except:
        return str(value)

def float_params(prec=None, binary=False, *argv):
    """
    Parameters
    -----------
    prec | str, 
        'q43', 'fp8-e4m3'         - NVIDIA quarter precision (4 exponent bits,
                                    3 significand bits)
        'q52', 'fp8-e5m2'         - NVIDIA quarter precision (5 exponent bits,
                                    2 significand bits)
        'b', 'bfloat16'           - bfloat16
        'h', 'half', 'fp16'       - IEEE half precision
        't', 'tf32'               - NVIDIA tf32
        's', 'single', 'fp32'     - IEEE single precision
        'd', 'double', 'fp64'     - IEEE double precision (the default)
        'q', 'quadruple', 'fp128' - IEEE quadruple precision

    For all these arithmetics the floating-point numbers have the form
    s * 2^e * d_0.d_1d_2...d_{t-1} where s = 1 or -1, e is the exponent
    and each d_i is 0 or 1, with d_0 = 1 for normalized numbers.
    With no input and output arguments, FLOAT_PARAMS prints a table showing
    all the parameters for all the precisions.
    Note: xmax and xmin are not representable in double precision for
   'quadruple'.

    Returns
    -----------
     u:     the unit roundoff,
     xmins: the smallest positive (subnormal) floating-point number,
     xmin:  the smallest positive normalized floating-point number,
     xmax:  the largest floating-point number,
     p:     the number of binary digits in the significand (including the
            implicit leading bit),
     emins  exponent of xmins,
     emin:  exponent of xmin,
     emax:  exponent of xmax.
    
    """                                                 
    if prec is None:
        precs = 'bhtsdq'
        
        data = pd.DataFrame(columns=['', 'u', 'xmins', 'xmin', 'xmax', 'p', 'emins', 'emin', 'emax'])
        for j in np.arange(-2, len(precs)):
            if j == -2:
                prec = 'q43'
                
            elif j == -1:
                prec = 'q52'
        
            else:
                prec = precs[j]
        
            (u, xmins, xmin, xmax, p, emins, emin, emax) = float_params(prec)
            
            if binary:
                data.loc[len(data.index)] = [f'{prec:s}', f'{binary_mark(u):s}', f'{binary_mark(xmins):s}',
                                             f'{binary_mark(xmin):s}', f'{binary_mark(xmax):s}', 
                                             f'{p:3.0f}', f'{emins:7.0f}', f'{emin:7.0f}', f'{emax:6.0f}']

            else:
                data.loc[len(data.index)] = [f'{prec:s}', f'{u:9.2e}',  f'{xmins:9.2e}',  f'{xmin:9.2e}',  f'{xmax:9.2e}',
                                             f'{p:3.0f}',  f'{emins:7.0f}',  f'{emin:7.0f}',  f'{emax:6.0f}']
        # print('-------------------------------------------------------------------------------')
        return data
    
    else:
            if prec in {'q43', 'fp8-e4m3'}:
                p = 4
                emax = 7
            elif prec in {'q52', 'fp8-e5m2'}:
                p = 3
                emax = 15
            elif prec in {'b', 'bfloat16'}:
                p = 8
                emax = 127  
            elif prec in {'h', 'half', 'fp16'}:
                p = 11
                emax = 15
            elif prec in {'t', 'tf32'}:
                p = 11
                emax = 127 
            elif prec in {'s', 'single', 'fp32'}:
                p = 24
                emax = 127
            elif prec in {'d', 'double', 'fp64'}:
                p = 53
                emax = 1023
            elif prec in {'q', 'quadruple', 'fp128'}:
                p = 113
                emax = 16383

            else:
                raise ValueError('Please specify a parameter supported by the software.')
                
        emin = 1 - emax
        emins = emin + 1 - p   
        xmins = 2**emins
        xmin = 2**emin
        
        try:
            xmax = 2**emax * (2-2**(1-p))
        except OverflowError:
            xmax = float('inf')
        
        u = 2**(-p)
        
        return u, xmins, xmin, xmax, p, emins, emin, emax


    
    
