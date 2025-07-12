from .simulate import Simulate

from .float_params import float_params
from .fixed_point import Chopf
from .integer import Chopi
from .chop import Chop
from .bitchop import Bitchop
from .lightchop import LightChop

__version__ = '0.3.3'  

import os
os.environ['chop_backend'] = 'numpy'
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