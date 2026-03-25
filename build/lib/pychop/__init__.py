"""
Pychop: Precision Simulation for Low-Precision Arithmetic

A comprehensive Python package for simulating low-precision arithmetic in
scientific computing and machine learning, with support for multiple backends
(NumPy, JAX, PyTorch).

Supported formats:
- Floating-point (Chop): IEEE 754 and custom formats
- Fixed-point (Chopf): Integer and fractional bits
- Integer quantization (Chopi): Symmetric and asymmetric
- Block Floating Point (BFP): Shared exponent per block
- Microscaling (MX): OCP standard with block-level scaling

Backends:
- NumPy: Pure numerical computation
- JAX: Custom VJP for differentiation
- PyTorch: Straight-Through Estimator (STE) for QAT

Author: Erin Carson, Xinye Chen
"""

from .chop import Chop
from .integer import Chopi
from .fixed_point import Chopf

from .simulate import Simulate

from .float_params import float_params
from .bitchop import Bitchop
from .faultchop import FaultChop

from .layers import ChopSTE, ChopfSTE, ChopiSTE
from .math_func import *


__version__ = '0.5.3'  

import os
if 'chop_backend' not in os.environ:
    os.environ['chop_backend'] = 'auto'
    
from .set_backend import backend
from .set_backend import get_backend

from dataclasses import dataclass
from typing import Optional

from .bfp_formats import (
    BFPSpec,
    BFPTensor,
    BFP_FORMATS,
    create_bfp_spec,
    bfp_quantize,
    print_bfp_format_table,
)

# MX Formats
from .mx_formats import (
    MXSpec,
    MXTensor,
    MX_FORMATS,
    create_mx_spec,
    mx_quantize,
    compare_mx_formats,
    print_mx_format_table,
)

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

