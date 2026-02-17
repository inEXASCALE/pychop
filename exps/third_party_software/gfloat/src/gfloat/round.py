# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Callable

import math

import numpy as np
import math

from .types import FormatInfo, RoundMode, Domain


def _isodd(v: int) -> bool:
    return v & 0x1 == 1


def _iseven(v: int) -> bool:
    return v & 0x1 == 0


def _rnitp(x: float, pred: Callable) -> int:
    """Round to nearest integer, ties to predicate"""
    floored = math.floor(x)
    should_round_away = (x > floored + 0.5) | ((x == floored + 0.5) & ~pred(floored))
    return floored + int(should_round_away)


def _rnite(x: float) -> int:
    """Round to nearest integer, ties to even"""
    return _rnitp(x, _iseven)


def _rnito(x: float) -> int:
    """Round to nearest integer, ties to odd"""
    return _rnitp(x, _isodd)


def round_float(
    fi: FormatInfo,
    v: float,
    rnd: RoundMode = RoundMode.TiesToEven,
    sat: bool = False,
    srbits: int = -1,
    srnumbits: int = 0,
) -> float:
    """
    Round input to the given :py:class:`FormatInfo`, given rounding mode and saturation flag

    An input NaN will convert to a NaN in the target.
    An input Infinity will convert to the largest float if :paramref:`sat`,
    otherwise to an Inf, if present, otherwise to a NaN.
    Negative zero will be returned if the format has negative zero, otherwise zero.

    Args:
      fi (FormatInfo): Describes the target format
      v (float): Input value to be rounded
      rnd (RoundMode): Rounding mode to use
      sat (bool): Saturation flag: if True, round overflowed values to `fi.max`
      srbits (int): Bits to use for stochastic rounding if rnd == Stochastic.
      srnumbits (int): How many bits are in srbits.  Implies srbits < 2**srnumbits.

    Returns:
      A float which is one of the values in the format.

    Raises:
       ValueError: The target format cannot represent the input
             (e.g. converting a `NaN`, or an `Inf` when the target has no
             `NaN` or `Inf`, and :paramref:`sat` is false)
       ValueError: Inconsistent arguments, e.g. srnumbits >= 2**srnumbits
    """

    # Constants
    p = fi.precision
    bias = fi.bias

    if rnd in (RoundMode.Stochastic, RoundMode.StochasticFast):
        if srbits >= 2**srnumbits:
            raise ValueError(f"srnumbits={srnumbits} >= 2**srnumbits={2**srnumbits}")

    if math.isnan(v):
        if fi.num_nans == 0:
            raise ValueError(f"No NaN in format {fi}")

        # Note that this does not preserve the NaN payload
        return np.nan

    # Extract sign
    sign = np.signbit([v]).item() and fi.is_signed
    vpos = -v if sign else v

    if math.isinf(vpos):
        result = np.inf

    elif vpos == 0:
        result = 0

    else:
        # Extract exponent
        expval = math.frexp(vpos)[1] - 1

        # Effective precision, accounting for right shift for subnormal values
        if fi.has_subnormals:
            expval = max(expval, 1 - bias)

        # Lift to "integer * 2^e"
        expval = expval - p + 1

        # use ldexp instead of vpos*2**-expval to avoid overflow
        fsignificand = math.ldexp(vpos, -expval)

        # Round
        isignificand = math.floor(fsignificand)
        delta = fsignificand - isignificand

        code_is_odd = (
            _isodd(isignificand)
            if fi.precision > 1
            else (isignificand != 0 and _isodd(expval + bias))
        )

        match rnd:
            case RoundMode.TowardZero:
                should_round_away = False
            case RoundMode.TowardPositive:
                should_round_away = not sign and delta > 0
            case RoundMode.TowardNegative:
                should_round_away = sign and delta > 0
            case RoundMode.TiesToAway:
                should_round_away = delta + 0.5 >= 1.0
            case RoundMode.TiesToEven:
                should_round_away = delta > 0.5 or (delta == 0.5 and code_is_odd)

            case RoundMode.StochasticFastest:
                assert srbits is not None
                exp2r = 2**srnumbits
                should_round_away = math.floor(delta * exp2r) + srbits >= exp2r

            case RoundMode.StochasticFast:
                assert srbits is not None
                exp2rp1 = 2 ** (1 + srnumbits)
                should_round_away = (
                    math.floor(delta * exp2rp1) + (2 * srbits + 1) >= exp2rp1
                )

            case RoundMode.Stochastic:
                assert srbits is not None
                exp2r = 2**srnumbits
                should_round_away = _rnite(delta * exp2r) + srbits >= exp2r

            case RoundMode.StochasticOdd:
                assert srbits is not None
                exp2r = 2**srnumbits
                should_round_away = _rnito(delta * exp2r) + srbits >= exp2r

        if should_round_away:
            # This may increase isignificand to 2**p,
            # which would require special casing in encode,
            # but not here, where we reconstruct a rounded value.
            isignificand += 1

        # Reconstruct rounded result to float
        result = isignificand * (2.0**expval)

    if result == 0:
        if sign and fi.has_nz:
            return -0.0
        else:
            return 0.0

    # Overflow
    amax = -fi.min if sign else fi.max
    if result > amax:
        if (
            sat
            or (rnd == RoundMode.TowardNegative and not sign and np.isfinite(v))
            or (rnd == RoundMode.TowardPositive and sign and np.isfinite(v))
            or (rnd == RoundMode.TowardZero and np.isfinite(v))
        ):
            result = amax
        else:
            if fi.domain == Domain.Extended:
                result = np.inf
            elif fi.num_nans > 0:
                result = np.nan
            else:
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    # Set sign
    if sign:
        result = -result

    return result
