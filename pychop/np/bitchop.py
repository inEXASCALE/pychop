"""NumPy bit-level floating-point chopping backend."""

import math

import numpy as np

from .._bitchop_common import (
    normalize_rmode, overflow_to_infinity, validate_format, validate_input_range,
)


class Bitchop:
    """Quantize float32 or float64 values into a configurable binary format."""

    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, random_state=42):
        validate_format(exp_bits, sig_bits)
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = normalize_rmode(rmode)
        self.subnormal = subnormal
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = 1 - self.bias
        self.mask_sig = (1 << sig_bits) - 1
        self.rng = np.random.RandomState(random_state)

    def __call__(self, values):
        self.value = np.asarray(values)
        self.dtype = self.value.dtype
        if self.dtype not in (np.float32, np.float64):
            raise ValueError("Input must be float32 or float64")
        validate_input_range(self.exp_bits, 8 if self.dtype == np.float32 else 11)

        self.sign = np.empty(self.value.shape, dtype=np.uint8)
        self.exponent = np.empty(self.value.shape, dtype=np.int64)
        self.significand = np.empty(self.value.shape, dtype=np.uint64)
        self.is_denormal = np.empty(self.value.shape, dtype=bool)
        self.rounding_value = np.empty_like(self.value)
        for index in np.ndindex(self.value.shape):
            result = self._quantize_scalar(self.value[index])
            self.sign[index], self.exponent[index], self.significand[index], self.is_denormal[index], self.rounding_value[index] = result
        return self.rounding_value

    def _round_up(self, quotient, remainder, shift, sign):
        if not remainder:
            return False
        if self.rmode in (0, 1):
            halfway = 1 << (shift - 1)
            return remainder > halfway or (remainder == halfway and (quotient & 1) != (self.rmode == 0))
        if self.rmode in (7, 8):
            halfway = 1 << (shift - 1)
            return remainder > halfway or (self.rmode == 8 and remainder == halfway)
        if self.rmode == 9:
            return not (quotient & 1)
        if self.rmode == 2:
            return not sign
        if self.rmode == 3:
            return bool(sign)
        if self.rmode == 4:
            return False
        if self.rmode == 5:
            return self.rng.random_sample() < math.ldexp(float(remainder), -shift)
        return self.rng.random_sample() < 0.5

    def _quantize_scalar(self, value):
        sign = int(np.signbit(value))
        signed_zero = math.copysign(0.0, -1.0 if sign else 1.0)
        special_exp = (1 << self.exp_bits) - 1
        if np.isnan(value):
            return sign, special_exp, 1, False, value
        if np.isinf(value):
            return sign, special_exp, 0, False, value
        if value == 0:
            return sign, 0, 0, False, signed_zero

        numerator, denominator = float(abs(value)).as_integer_ratio()
        source_power = -(denominator.bit_length() - 1)
        exponent = numerator.bit_length() - 1 + source_power

        def overflow():
            if overflow_to_infinity(self.rmode, sign):
                return sign, special_exp, 0, False, math.copysign(math.inf, value)
            max_value = math.ldexp(2.0 - math.ldexp(1.0, -self.sig_bits), self.max_exp)
            return sign, self.max_exp + self.bias, self.mask_sig, False, math.copysign(max_value, value)

        if exponent > self.max_exp:
            return overflow()

        scale_exp = max(exponent, self.min_exp)
        quantum_power = scale_exp - self.sig_bits
        shift = quantum_power - source_power
        if shift > 0:
            quotient, remainder = divmod(numerator, 1 << shift)
            quotient += self._round_up(quotient, remainder, shift, sign)
        else:
            quotient = numerator << -shift

        if quotient >= (1 << (self.sig_bits + 1)):
            quotient >>= 1
            scale_exp += 1
        if scale_exp > self.max_exp:
            return overflow()
        if quotient == 0:
            return sign, 0, 0, False, signed_zero

        is_denormal = quotient < (1 << self.sig_bits)
        if is_denormal and not self.subnormal:
            return sign, 0, 0, False, signed_zero
        biased_exp = 0 if is_denormal else scale_exp + self.bias
        sig_int = quotient if is_denormal else quotient - (1 << self.sig_bits)
        result = math.ldexp(float(quotient), scale_exp - self.sig_bits)
        return sign, biased_exp, sig_int, is_denormal, math.copysign(result, value)

    def __str__(self):
        lines = []
        for value, sign, exponent, significand, denormal, result in zip(
            self.value.flat, self.sign.flat, self.exponent.flat,
            self.significand.flat, self.is_denormal.flat, self.rounding_value.flat,
        ):
            lines.append(
                f"value: {value}, sign: {sign}, exp: {exponent}, "
                f"sig: {int(significand):0{self.sig_bits}b}, "
                f"emulated value: {result}, denorm: {denormal}"
            )
        return "\n".join(lines)
