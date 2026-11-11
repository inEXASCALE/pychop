"""Shared format and rounding rules for Bitchop backends."""

from numbers import Integral

ROUNDING_MODES = {
    "nearest_odd": 0,
    "nearest_even": 1,
    "plus_infinity": 2,
    "minus_infinity": 3,
    "toward_zero": 4,
    "stochastic_prop": 5,
    "stochastic_equal": 6,
}


def normalize_rmode(rmode):
    if isinstance(rmode, str):
        try:
            return ROUNDING_MODES[rmode]
        except KeyError:
            pass
    elif isinstance(rmode, Integral) and not isinstance(rmode, bool) and 0 <= rmode <= 6:
        return int(rmode)
    raise ValueError(f"Unknown rounding mode: {rmode!r}")


def validate_format(exp_bits, sig_bits):
    if not isinstance(exp_bits, Integral) or isinstance(exp_bits, bool) or exp_bits < 2:
        raise ValueError("exp_bits must be an integer of at least 2")
    if not isinstance(sig_bits, Integral) or isinstance(sig_bits, bool) or not 0 <= sig_bits <= 52:
        raise ValueError("sig_bits must be an integer between 0 and 52")


def validate_input_range(exp_bits, source_exp_bits):
    if exp_bits > source_exp_bits:
        raise ValueError(
            "Target exponent width exceeds the input dtype's range; "
            "the output cannot be represented in the input dtype"
        )


def overflow_to_infinity(rmode, sign):
    return rmode in (0, 1, 5, 6) or (rmode == 2 and not sign) or (rmode == 3 and sign)
