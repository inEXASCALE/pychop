# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Callable
import ml_dtypes
import numpy as np
import pytest

from gfloat import FloatClass, Domain, decode_float, decode_ndarray
from gfloat.formats import *


def spec_is_normal(fi: FormatInfo, x: int) -> bool:
    r"""
  Copy from spec:

  \Case{\isNormal*<f>(x \in \{0, \NInf, \Inf, \NaN\}) \gives \False}\\
  \Case{\isNormal*<f>(x) \gives
    \begin{cases}
      (x \mod 2^{\k_f - 1}) \div 2^{\p_f - 1} > 0 & \If \s_f = \Signed   \\
      x \div 2^{\p_f - 1} > 0                     & \If \s_f = \Unsigned
    \end{cases}
  }
  """
    if x in (fi.code_of_zero, fi.code_of_nan):
        return False
    if fi.num_neginfs > 0 and x == fi.code_of_neginf:
        return False
    if fi.num_posinfs > 0 and x == fi.code_of_posinf:
        return False

    k_f = fi.k
    p_f = fi.precision
    if fi.is_signed:
        #      (x \mod 2^{\k_f - 1}) \div 2^{\p_f - 1} > 0
        return (x % 2 ** (k_f - 1)) // 2 ** (p_f - 1) > 0
    else:
        #      x \div 2^{\p_f - 1} > 0
        return x // 2 ** (p_f - 1) > 0


_p3109_formats_to_test = (
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (6, 1),
    (6, 5),
    (8, 3),
    (8, 1),
    (11, 3),
)


@pytest.mark.parametrize("k,p", _p3109_formats_to_test)
@pytest.mark.parametrize("signedness", Signedness)
def test_p3109_specials_signed(k: int, p: int, signedness: Signedness) -> None:
    fi = format_info_p3109(k, p, signedness, Domain.Extended)

    for i in range(2**fi.k):
        fv = decode_float(fi, i)
        assert spec_is_normal(fi, i) == (fv.fclass == FloatClass.NORMAL)
