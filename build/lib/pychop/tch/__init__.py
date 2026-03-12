from .integer import *
from .fixed_point import *
from .bitchop import Bitchop
from .float_point import Chop_
from .lightchop import LightChop_

try:
    from . import layers
    _LAYERS_AVAILABLE = True
except ImportError as e:
    if 'torch' in str(e):
        layers = None
        _LAYERS_AVAILABLE = False
    else:
        raise

__all__ = [
    'Bitchop', 'Chop_', 'LightChop_',
]

if _LAYERS_AVAILABLE:
    __all__.append('layers')