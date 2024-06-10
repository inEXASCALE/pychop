import numpy as np

def round_clamp(x, bits=8):
    x = np.clip(x, a_min=0, a_max=2**(bits)-1)
    x = np.round(x * 2**(bits)) / (2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    x_f = np.sign(x)*round_clamp(np.abs(x) - np.floor(np.abs(x)), fbits)
    x_i = np.sign(x)*round_clamp(np.floor(np.abs(x)), ibits)
    return (x_i + x_f)