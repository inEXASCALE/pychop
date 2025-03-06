from .simulate import simulate
from .set_backend import backend
from .float_params import float_params
from .fixed_point import FPoint
from .chop import Chop
from .bitchop import Bitchop
from .quant import quant
from .iqlayer import IntQuantizedLayer
from .qlayer import QuantizedLayer
from .qlayer import Rounding as Round

__version__ = '0.3.1'  
backend('numpy')

from dataclasses import dataclass
from typing import Optional

@dataclass
class Customs:
    emax: Optional[int] = None
    t: Optional[int] = None
    exp_bits: Optional[int] = None
    sig_bits: Optional[int] = None

@dataclass
class Options:
    t: int
    emax: int
    prec: int
    subnormal: bool
    rmode: bool
    flip: bool
    explim: bool
    p: float