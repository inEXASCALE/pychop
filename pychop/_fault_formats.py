"""Named format settings shared by the FaultChop backends."""

from copy import copy
from numbers import Integral


# (significand precision including the hidden bit, maximum normal exponent)
FORMAT_PARAMS = {
    "q43": (4, 7),  # MATLAB-compatible, IEEE-style E4M3
    "fp8-e4m3": (4, 8),  # OCP E4M3; its top code is reserved for NaN
    "q52": (3, 15),
    "fp8-e5m2": (3, 15),
    "h": (11, 15), "half": (11, 15), "fp16": (11, 15),
    "b": (8, 127), "bfloat16": (8, 127), "bf16": (8, 127),
    "t": (11, 127), "tf32": (11, 127),
    "s": (24, 127), "single": (24, 127), "fp32": (24, 127),
    "d": (53, 1023), "double": (53, 1023), "fp64": (53, 1023),
}

NO_SUBNORMAL_DEFAULT = frozenset(("b", "bfloat16", "bf16"))


def exponent_bits_from_emax(emax):
    """Return the IEEE-style exponent width, or None for another range."""
    width = (emax + 1).bit_length()
    return width if (1 << (width - 1)) - 1 == emax else None


def resolve_customs(customs):
    """Resolve equivalent format fields without modifying the caller's object."""
    resolved = copy(customs)
    exp_bits = getattr(customs, "exp_bits", None)
    sig_bits = getattr(customs, "sig_bits", None)
    emax = getattr(customs, "emax", None)
    t = getattr(customs, "t", None)

    if exp_bits is not None:
        if not isinstance(exp_bits, Integral) or isinstance(exp_bits, bool) or exp_bits < 2:
            raise ValueError("exp_bits must be an integer of at least 2")
        derived_emax = (1 << (exp_bits - 1)) - 1
        if emax is not None and emax != derived_emax:
            raise ValueError("emax conflicts with exp_bits")
        emax = derived_emax
    if sig_bits is not None:
        if not isinstance(sig_bits, Integral) or isinstance(sig_bits, bool) or sig_bits < 1:
            raise ValueError("sig_bits must be a positive integer")
        derived_t = sig_bits + 1
        if t is not None and t != derived_t:
            raise ValueError("t conflicts with sig_bits")
        t = derived_t
    if not isinstance(emax, Integral) or emax < 1:
        raise ValueError("Customs requires emax or exp_bits")
    if not isinstance(t, Integral) or t < 2:
        raise ValueError("Customs requires t or sig_bits")
    resolved.emax = int(emax)
    resolved.t = int(t)
    return resolved
