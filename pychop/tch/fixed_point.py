import torch

def round_clamp(x, bits=8):
    x = x.clamp(0,2**(bits)-1)
    x = x.mul(2**(bits)).round().div(2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    x_f = x.sign()*round_clamp(torch.abs(x) - torch.abs(x).floor(), fbits)
    x_i = x.sign()*round_clamp(x.abs().floor(), ibits)
    return (x_i + x_f)
