import os

_VALID_BACKENDS = ('numpy', 'jax', 'torch', 'auto')


def backend(lib='auto', verbose=0):
    """
    Set the compute backend for pychop.

    Parameters
    ----------
    lib : str
        The backend library. One of 'numpy', 'jax', 'torch', or 'auto'.
        'auto' detects the backend from the input array type at runtime.

    verbose : int | bool
        Whether or not to print the information.

    Raises
    ------
    ValueError
        If `lib` is not a supported backend name.

    Examples
    --------
    >>> import pychop
    >>> pychop.backend('jax')
    >>> pychop.backend('numpy')
    >>> pychop.backend('torch')
    >>> pychop.backend('auto')   # detect from input array type
    """
    if lib not in _VALID_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{lib}'. "
            f"Must be one of: {_VALID_BACKENDS}"
        )

    os.environ['chop_backend'] = lib

    if lib == 'numpy':
        try:
            import numpy  # noqa: F401 — verify it's importable
            if verbose:
                print('Load NumPy backend.')
        except ImportError as e:
            raise ImportError('NumPy is not installed.') from e

    elif lib == 'jax':
        try:
            import jax  # noqa: F401 — verify it's importable
            if verbose:
                print('Load JAX backend.')
        except ImportError as e:
            print(f'JAX is not installed ({e}). Falling back to NumPy backend.')
            os.environ['chop_backend'] = 'numpy'

    elif lib == 'torch':
        try:
            import torch  
            if verbose:
                print('Load Torch backend.')
        except ImportError as e:
            print(f'PyTorch is not installed ({e}). Falling back to NumPy backend.')
            os.environ['chop_backend'] = 'numpy'

    elif lib == 'auto':
        if verbose:
            print('Backend set to auto: will be detected from input array type at runtime.')


def get_backend():
    """
    Return the currently active backend name.

    Returns
    -------
    str
        One of 'numpy', 'jax', 'torch', or 'auto'.
    """
    return os.environ.get('chop_backend', 'auto')
