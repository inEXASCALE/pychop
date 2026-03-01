from .chop import Chop
from .integer import Chopi
from .fixed_point import Chopf

from .simulate import Simulate

from .float_params import float_params
from .bitchop import Bitchop
from .faultchop import FaultChop

from .layers import ChopSTE, ChopfSTE, ChopiSTE
from .math_func import *


__version__ = '0.4.6'  

import os
os.environ['chop_backend'] = 'auto'
from .set_backend import backend


from dataclasses import dataclass
from typing import Optional

@dataclass
class Customs:
    emax: Optional[int] = None # the maximum value of the exponent.
    t: Optional[int] = None # the number of bits in the significand (including the hidden bit)
    exp_bits: Optional[int] = None # the exponent bits
    sig_bits: Optional[int] = None  # the significand bits (not including the hidden bit)


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

