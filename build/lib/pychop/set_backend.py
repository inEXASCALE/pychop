import os


def backend(lib='numpy', verbose=0):
    """
    Parameters
    ----------
    lib : str,
        The backend library.    
    
    verbose : int | bool
        Whether or not to print the information.

    """
    os.environ['chop_backend'] = lib

    if lib == 'numpy':
        try:
            global numpy
            import numpy 
            if verbose: print('Load NumPy backend.')
        except ImportError as e:
            print(e, 'Fail to load NumPy backend.')

    elif lib == 'jax':
        try:
            global jax
            import jax
            if verbose: print('Load JAX backend.')
        except ImportError as e:
            print(e, 'Load NumPy backend.')
            backend('numpy')
    else:
        try:
            global torch
            global FQuantizedLayer
            global IntQuantizedLayer
            global QuantizedLayer
            
            import torch
            if verbose: print('Load Torch backend.')
                
        except ImportError as e:
            print(e, 'Try load NumPy backend.')
            backend('numpy')


if os.environ['chop_backend'] == "jax":
    global jax

elif os.environ['chop_backend'] == "torch":
    global torch
    global LightChop
    global FQuantizedLayer
    global IntQuantizedLayer
    global QuantizedLayer
    
