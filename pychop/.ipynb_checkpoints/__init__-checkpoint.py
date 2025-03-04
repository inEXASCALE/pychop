from .simulate import simulate
from .np.chop import customs
from .set_backend import backend
from .float_params import float_params
from .fixed_point import fpoint
from .chop import chop
from .bitchop import bitchop
from .quant import quant
from .iqlayer import IntQuantizedLayer
from .qlayer import QuantizedLayer, Rounding

__version__ = '0.3.1'  
backend('numpy')
