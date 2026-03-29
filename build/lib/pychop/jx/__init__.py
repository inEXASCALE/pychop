from .integer import *
from .fixed_point import *
from .bitchop import Bitchop
from .float_point import Chop_
from .lightchop import LightChop_

_layers = None


def __getattr__(name):
    """Lazy import for layers module.
    
    This ensures that flax is only required when actually using
    quantized layers, not when just importing the JAX backend.
    """
    global _layers
    if name == 'layers':
        if _layers is None:
            try:
                from . import layers as _layers_module
                _layers = _layers_module
            except ImportError as e:
                if 'flax' in str(e):
                    raise ImportError(
                        "JAX quantized layers require 'flax' to be installed. "
                        "Install it with: pip install flax"
                    ) from e
                raise
        return _layers
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'Bitchop', 'Chop_', 'LightChop_',
    'layers',  # Will be lazily loaded
]