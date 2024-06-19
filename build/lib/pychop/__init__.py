from .simulate import simulate
from .np.chop import customs
from .set_backend import backend
from .float_params import float_params
from .fixed_point import fpoint
from .chop import chop
from .quant import quant
from .qtrain import QuantLayer

__version__ = '0.2.6'  
backend('numpy')