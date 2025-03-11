import os

def LightChop(exp_bits: int, sig_bits: int, rmode: int = 1, random_state: int=42):
    """
    A class to simulate different floating-point precisions and rounding modes
    for PyTorch tensors.
    """
    if os.environ['chop_backend'] == 'torch':
        from .tch.lightchop import LightChop
    
    elif os.environ['chop_backend'] == 'jax':
        from .jx.lightchop import LightChop
        
    else:
        from .np.lightchop import LightChop

    return LightChop(exp_bits, sig_bits, rmode, random_state)
