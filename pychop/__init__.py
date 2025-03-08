from .simulate import Simulate

from .float_params import float_params
from .fixed_point import Chopf
from .integer import Chopi
from .chop import Chop
from .bitchop import Bitchop
from .integer import quant



__version__ = '0.3.1'  

import os
os.environ['chop_backend'] = 'numpy'
from .set_backend import backend


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