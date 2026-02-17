# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from dataclasses import dataclass
from enum import Enum
import math


class RoundMode(Enum):
    """
    Enum for IEEE-754 rounding modes.

    Result :math:`r` is obtained from input :math:`v` depending on rounding mode as follows

    Notes on stochastic rounding:

    StochasticFast implements a stochastic rounding scheme that is unbiased in
    infinite precision, but biased when the quantity to be rounded is computed to
    a finite precision.

    StochasticFastest implements a stochastic rounding scheme that is biased
    (the rounded value is on average farther from zero than the true value).

    With a lot of SRbits (say 8 or more), these biases are negligible, and there
    may be some efficiency advantage in using StochasticFast or StochasticFastest.

    """

    TowardZero = 1  #: Return the largest :math:`r` such that :math:`|r| \le |v|`
    TowardNegative = 2  #: Return the largest :math:`r` such that :math:`r \le v`
    TowardPositive = 3  #: Return the smallest :math:`r` such that :math:`r \ge v`
    TiesToEven = 4  #: Round to nearest, ties to even
    TiesToAway = 5  #: Round to nearest, ties away from zero
    Stochastic = 6  #: Stochastic rounding, RTNE before comparison
    StochasticOdd = 7  #: Stochastic rounding, RTNO before comparison
    StochasticFast = 8  #: Stochastic rounding - faster, but biased
    StochasticFastest = 9  #: Stochastic rounding - even faster, but more biased


class Domain(Enum):
    """
    Enum for domain of values
    """

    Finite = 1  #: Finite values only
    Extended = 2  #: Finite values and infinities


class Signedness(Enum):
    """
    Enum for domain of values
    """

    Signed = 1  #: Positive and negative values
    Unsigned = 2  #: Positive values only


class FloatClass(Enum):
    """
    Enum for the classification of a FloatValue.
    """

    NORMAL = 1  #: A positive or negative normalized non-zero value
    SUBNORMAL = 2  #: A positive or negative subnormal value
    ZERO = 3  #: A positive or negative zero value
    INFINITE = 4  #: A positive or negative infinity (+/-Inf)
    NAN = 5  #: Not a Number (NaN)


@dataclass
class FloatValue:
    """
    A floating-point value decoded in great detail.
    """

    code: int  #: Integer code point

    #: Value. Assumed to be exactly round-trippable to python float.
    #: This is true for all <64bit formats known in 2023.
    fval: float

    exp: int  #: Raw exponent without bias
    expval: int  #: Exponent, bias subtracted
    significand: int  #: Significand as an integer
    fsignificand: float  #: Significand as a float in the range [0,2)
    signbit: int  #: Sign bit: 1 => negative, 0 => positive
    fclass: FloatClass  #: See FloatClass


@dataclass
class FormatInfo:
    """
    Class describing a floating-point format, parametrized
    by width, precision, and special value encoding rules.

    """

    #: Short name for the format, e.g. binary32, bfloat16
    name: str

    #: Number of bits in the format
    k: int

    #: Number of significand bits (including implicit leading bit)
    precision: int

    #: Exponent bias
    bias: int

    #: Signedness: True if the format encodes negative numbers
    is_signed: bool

    #: Domain: Finite or Extended
    #: If Extended, the non-nan value with the highest encoding for each sign (s)
    #: is replaced by (s)Inf.
    domain: Domain

    #: Set if format encodes -0 at (sgn=1,exp=0,significand=0).
    #: If False, that encoding decodes to a NaN labelled NaN_0
    has_nz: bool

    #: Number of NaNs that are encoded in the highest encodings for each sign
    num_high_nans: int

    #: Set if format encodes subnormals
    has_subnormals: bool

    #: Set if the format uses two's complement encoding for the significand
    is_twos_complement: bool

    def __init__(
        self,
        name: str,
        k: int,
        precision: int,
        *,
        bias: int,
        is_signed: bool,
        domain: Domain,
        has_nz: bool,
        num_high_nans: int,
        has_subnormals: bool,
        is_twos_complement: bool,
    ):
        self.name = name
        self.k = k
        self.precision = precision
        self.bias = bias
        self.is_signed = is_signed
        self.domain = domain
        self.has_nz = has_nz
        self.num_high_nans = num_high_nans
        self.has_subnormals = has_subnormals
        self.is_twos_complement = is_twos_complement

    #: ## Derived values

    @property
    def tSignificandBits(self) -> int:
        """The number of trailing significand bits, t"""
        return self.precision - 1

    @property
    def expBits(self) -> int:
        """The number of exponent bits, w"""
        return self.k - self.tSignificandBits - self.signBits

    @property
    def signBits(self) -> int:
        """The number of sign bits, s"""
        return 1 if self.is_signed else 0

    @property
    def emax(self) -> int:
        """Return
        :math:`floor(log_2(maxFinite)) = emax`
        Note that for an all-subnormal format, this is not necessarily the
        largest value in the exponent field.
        """
        return math.floor(math.log2(self.max))

    # numpy finfo properties
    @property
    def bits(self) -> int:
        """
        The number of bits occupied by the type.
        """
        return self.k

    # @property
    # def dtype(self) -> np.dtype:
    #     """
    #     Returns the dtype for which `finfo` returns information. For complex
    #     input, the returned dtype is the associated ``float*`` dtype for its
    #     real and complex components.
    #     """

    @property
    def eps(self) -> float:
        """
        The difference between 1.0 and the smallest representable float
        larger than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``eps = 2**-52``, approximately 2.22e-16.
        """
        # TODO: Check if 1.0 is subnormal for any reasonable format, e.g. p3109(7)?
        return 2**self.machep

    @property
    def epsneg(self) -> float:
        """
        The difference between 1.0 and the largest representable float
        less than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``epsneg = 2**-53``, approximately 1.11e-16.
        """
        return self.eps / 2

    @property
    def iexp(self) -> int:
        """
        The number of bits in the exponent portion of the floating point
        representation.
        """
        return self.expBits

    @property
    def machep(self) -> int:
        """
        The exponent that yields `eps`.
        """
        return -self.tSignificandBits

    @property
    def max(self) -> float:
        """
        The largest representable number.
        """
        num_non_finites = self.num_high_nans + self.num_posinfs
        if num_non_finites == 2**self.tSignificandBits:
            # All-bits-one exponent field is full, value is in the
            # binade below, so significand is 0xFFF..F
            isig = 2**self.tSignificandBits - 1
            emax = 2**self.expBits - 2
        elif num_non_finites == 2 ** (self.tSignificandBits + 1):
            # Top two binades are full, value is in the
            # binade below them. Significand is still 0xFFF..F
            isig = 2**self.tSignificandBits - 1
            emax = 2**self.expBits - 3
        else:
            assert num_non_finites < 2**self.tSignificandBits
            # All-bits-one exponent field is not full, value is in the
            # final binade, so significand is 0xFFF..F - num_non_finites
            isig = 2**self.tSignificandBits - 1 - num_non_finites
            emax = 2**self.expBits - 1

        if self.is_all_subnormal:
            return 2 ** (emax - self.bias) * (isig * 2 ** (1 - self.tSignificandBits))
        else:
            return 2 ** (emax - self.bias) * (1.0 + isig * 2**-self.tSignificandBits)

    @property
    def maxexp(self) -> int:
        """
        The smallest positive power of the base (2) that causes overflow.
        """
        return self.emax + 1

    @property
    def min(self) -> float:
        """
        The smallest representable number, typically ``-max``.
        """
        if self.is_signed:
            if not self.is_twos_complement:
                return -self.max
            else:
                assert (
                    (self.domain == Domain.Finite)
                    and (self.num_high_nans == 0)
                    and not self.has_nz
                )
                return -(2.0 ** (self.emax + 1))
        else:
            # Unsigned
            if self.has_zero:
                return 0.0
            else:
                return 2**-self.bias

    @property
    def num_nans(self) -> int:
        """
        The number of code points which decode to NaN
        """
        if not self.is_signed:
            return self.num_high_nans

        # Signed
        if self.is_twos_complement:
            assert (
                (self.domain == Domain.Finite)
                and (self.num_high_nans == 0)
                and not self.has_nz
            )
            return 0

        return (0 if self.has_nz else 1) + 2 * self.num_high_nans

    @property
    def code_of_nan(self) -> int:
        """
        Return a codepoint for a NaN
        """
        if self.num_high_nans > 0:
            return 2 ** (self.k) - 1
        if not self.has_nz:
            return 2 ** (self.k - 1)
        raise ValueError(f"No NaN in {self}")

    @property
    def code_of_posinf(self) -> int:
        """
        Return a codepoint for positive infinity
        """
        if self.domain != Domain.Extended:
            raise ValueError(f"No Inf in {self}")

        return 2 ** (self.k - self.signBits) - 1 - self.num_high_nans

    @property
    def code_of_neginf(self) -> int:
        """
        Return a codepoint for negative infinity
        """
        if not (self.domain == Domain.Extended and self.is_signed):
            raise ValueError(f"No -Inf in {self}")

        return 2**self.k - 1 - self.num_high_nans

    @property
    def code_of_zero(self) -> int:
        """
        Return a codepoint for (non-negative) zero
        """
        assert self.has_zero
        return 0

    @property
    def has_zero(self) -> bool:
        """
        Does the format have zero?

        This is false if the mantissa is 0 width and we don't have subnormals -
        essentially the mantissa is always decoded as 1.
        If we have subnormals, the only subnormal is zero, and the mantissa is
        always decoded as 0.
        """
        return self.precision > 1 or self.has_subnormals

    @property
    def code_of_negzero(self) -> int:
        """
        Return a codepoint for negative zero
        """
        if not self.has_nz:
            raise ValueError(f"No negative zero in {self}")

        return 2 ** (self.k - 1)

    @property
    def num_posinfs(self) -> int:
        """
        Return the number of positive infinities
        """
        return 1 if self.domain == Domain.Extended else 0

    @property
    def num_neginfs(self) -> int:
        """
        Return the number of negative infinities
        """
        return 1 if self.domain == Domain.Extended and self.is_signed else 0

    @property
    def num_infs(self) -> int:
        """
        Return the number of infinities
        """
        return self.num_posinfs + self.num_neginfs

    @property
    def code_of_max(self) -> int:
        """
        Return a codepoint for fi.max
        """
        return 2 ** (self.k - self.signBits) - 1 - self.num_high_nans - self.num_posinfs

    @property
    def code_of_min(self) -> int:
        """
        Return a codepoint for fi.min
        """
        if self.is_signed and not self.is_twos_complement:
            return 2**self.k - self.num_high_nans - self.num_posinfs - 1
        elif self.is_signed and self.is_twos_complement:
            return 2 ** (self.k - 1)
        else:
            return 0  # codepoint of smallest value, whether 0 or 2^-bias

    # @property
    # def minexp(self) -> int:
    #     """
    #     The most negative power of the base (2) consistent with there
    #     being no leading 0's in the mantissa.
    #     """

    # @property
    # def negep(self) -> int:
    #     """
    #     The exponent that yields `epsneg`.
    #     """

    # @property
    # def nexp(self) -> int:
    #     """
    #     The number of bits in the exponent including its sign and bias.
    #     """

    # @property
    # def nmant(self) -> int:
    #     """
    #     The number of bits in the mantissa.
    #     """

    # @property
    # def precision(self) -> int:
    #     """
    #     The approximate number of decimal digits to which this kind of
    #     float is precise.
    #     """

    # @property
    # def resolution(self) -> float:
    #     """
    #     The approximate decimal resolution of this type, i.e.,
    #     ``10**-precision``.
    #     """

    # @property
    # def tiny(self) -> float:
    #     """
    #     An alias for `smallest_normal`, kept for backwards compatibility.
    #     """

    @property
    def smallest_normal(self) -> float:
        """
        The smallest positive floating point number with 1 as leading bit in
        the significand following IEEE-754.
        """
        if self.has_subnormals:
            return 2 ** (1 - self.bias)
        elif self.has_zero:
            return 2**-self.bias + 2 ** (-self.bias - self.tSignificandBits)
        else:
            return 2**-self.bias

    @property
    def smallest_subnormal(self) -> float:
        """
        The smallest positive floating point number with 0 as leading bit in
        the significand following IEEE-754.
        """
        assert self.has_subnormals, "not implemented"
        return 2 ** -(self.bias + self.tSignificandBits - 1)

    @property
    def smallest(self) -> float:
        """
        The smallest positive floating point number.
        """
        if self.has_subnormals:
            return self.smallest_subnormal
        else:
            return self.smallest_normal

    @property
    def is_all_subnormal(self) -> bool:
        """
        Are all encoded values subnormal?
        """
        return (self.expBits == 0) and self.has_subnormals

    @property
    def __name__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"{self.name}"
