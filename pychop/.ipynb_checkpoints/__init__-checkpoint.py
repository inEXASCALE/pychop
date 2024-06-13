from .simulate import simulate
from .np.chop import customs
from .set_backend import backend
from .float_params import float_params
from .fixed_point import to_fixed_point
from .chop import chop
from .quant import quant


__version__ = '0.2.4'  
backend('numpy')