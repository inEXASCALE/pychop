from .simulate import simulate
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

from dataclasses import dataclass
from typing import Optional

@dataclass
class customs:
    t: Optional[int] = None
    emax: Optional[int] = None
    exp_bits: int = None
    ig_bits: int = None

@dataclass
class options:
    t: int
    emax: int
    prec: int
    subnormal: bool
    rmode: bool
    flip: bool
    explim: bool
    p: float