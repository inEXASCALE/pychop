"""PyTorch bit-level floating-point chopping backend."""

import torch

from .._bitchop_common import (
    normalize_rmode, overflow_to_infinity, validate_format, validate_input_range,
)


class Bitchop:
    """Quantize float32 or float64 tensors into a configurable binary format."""

    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, random_state=42, device="cpu"):
        validate_format(exp_bits, sig_bits)
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = normalize_rmode(rmode)
        self.subnormal = subnormal
        self.device = device
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = 1 - self.bias
        self.mask_sig = (1 << sig_bits) - 1
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(random_state)

    def __call__(self, values):
        self.value = torch.as_tensor(values, device=self.device)
        self.dtype = self.value.dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Input must be torch.float32 or torch.float64")
        validate_input_range(self.exp_bits, 8 if self.dtype == torch.float32 else 11)
        if self.value.ndim == 0:
            self.value = self.value.unsqueeze(0)
        self.sign, self.exponent, self.significand, self.is_denormal, self.rounding_value = self._convert()
        return self.rounding_value

    def _convert(self):
        value = self.value
        if self.dtype == torch.float32:
            bits = value.view(torch.int32).to(torch.int64)
            source_sig_bits, source_bias, source_min_exp = 23, 127, -126
            raw_exp = (bits >> 23) & 0xff
        else:
            bits = value.view(torch.int64)
            source_sig_bits, source_bias, source_min_exp = 52, 1023, -1022
            raw_exp = (bits >> 52) & 0x7ff

        sign = (bits >> (source_sig_bits + (8 if self.dtype == torch.float32 else 11))) & 1
        fraction = bits & ((1 << source_sig_bits) - 1)
        special = raw_exp == (2 * source_bias + 1)
        zero = (raw_exp == 0) & (fraction == 0)
        source_power = torch.where(raw_exp == 0, source_min_exp, raw_exp - source_bias) - source_sig_bits
        whole_sig = torch.where(raw_exp == 0, fraction, fraction | (1 << source_sig_bits))
        # The fraction is an exact integer even when the source value is subnormal.
        _, fraction_exp = torch.frexp(fraction.clamp(min=1).to(torch.float64))
        actual_exp = torch.where(
            raw_exp == 0,
            fraction_exp.to(torch.int64) - 1 + source_min_exp - source_sig_bits,
            raw_exp - source_bias,
        )
        scale_exp = torch.maximum(actual_exp, torch.full_like(actual_exp, self.min_exp))
        shift = scale_exp - self.sig_bits - source_power

        right_shift = shift.clamp(0, 63)
        left_shift = (-shift).clamp(0, 52)
        right_quotient = whole_sig >> right_shift
        quotient = torch.where(shift > 0, right_quotient, whole_sig << left_shift)
        remainder = torch.where(shift > 0, whole_sig - (right_quotient << right_shift), 0)
        inexact = remainder != 0
        halfway = torch.ones_like(remainder) << (shift - 1).clamp(0, 62)

        if self.rmode in (0, 1):
            odd = (quotient & 1) != 0
            tie_up = odd if self.rmode == 1 else ~odd
            round_up = (remainder > halfway) | ((remainder == halfway) & tie_up)
            round_up &= inexact
        elif self.rmode == 2:
            round_up = inexact & (sign == 0)
        elif self.rmode == 3:
            round_up = inexact & (sign == 1)
        elif self.rmode == 4:
            round_up = torch.zeros_like(inexact)
        else:
            random_value = torch.rand(value.shape, generator=self.rng, device=value.device, dtype=value.dtype)
            if self.rmode == 5:
                probability = torch.ldexp(remainder.to(value.dtype), -shift.to(torch.int32))
                round_up = inexact & (random_value < probability)
            else:
                round_up = inexact & (random_value < 0.5)

        rounded = quotient + round_up.to(torch.int64)
        carry = rounded >= (1 << (self.sig_bits + 1))
        rounded = torch.where(carry, rounded >> 1, rounded)
        scale_exp = scale_exp + carry.to(torch.int64)

        overflow = scale_exp > self.max_exp
        flushed = (rounded < (1 << self.sig_bits)) & (not self.subnormal) & ~overflow & ~special
        is_denormal = (rounded > 0) & (rounded < (1 << self.sig_bits)) & ~flushed & ~overflow & ~special
        zero_result = zero | flushed | ((rounded == 0) & ~overflow & ~special)
        biased_exp = torch.where(is_denormal | zero_result, 0, scale_exp + self.bias)
        sig_int = torch.where(is_denormal, rounded, rounded - (1 << self.sig_bits))

        magnitude = torch.ldexp(rounded.to(value.dtype), (scale_exp - self.sig_bits).to(torch.int32))
        finite = torch.copysign(magnitude, value)
        if overflow_to_infinity(self.rmode, False) and overflow_to_infinity(self.rmode, True):
            overflow_value = torch.copysign(torch.full_like(value, torch.inf), value)
        else:
            max_value = torch.ldexp(
                torch.full_like(value, 2.0 - 2.0 ** -self.sig_bits),
                torch.full_like(scale_exp, self.max_exp, dtype=torch.int32),
            )
            infinity = torch.full_like(value, torch.inf)
            to_inf = (sign == 0) if self.rmode == 2 else (sign == 1) if self.rmode == 3 else torch.zeros_like(zero)
            overflow_value = torch.copysign(torch.where(to_inf, infinity, max_value), value)
        result = torch.where(overflow, overflow_value, finite)
        result = torch.where(zero_result, torch.copysign(torch.zeros_like(value), value), result)
        result = torch.where(special, value, result)

        special_exp = (1 << self.exp_bits) - 1
        overflow_inf = overflow & torch.isinf(result)
        biased_exp = torch.where(overflow_inf, special_exp, biased_exp)
        biased_exp = torch.where(overflow & ~overflow_inf, self.max_exp + self.bias, biased_exp)
        biased_exp = torch.where(special, special_exp, biased_exp)
        sig_int = torch.where(zero_result | overflow_inf, 0, sig_int)
        sig_int = torch.where(overflow & ~overflow_inf, self.mask_sig, sig_int)
        sig_int = torch.where(special & torch.isnan(value), 1, sig_int)
        sig_int = torch.where(special & torch.isinf(value), 0, sig_int)
        return sign.to(torch.uint8), biased_exp.to(torch.int64), sig_int.to(torch.int64), is_denormal, result

    def __str__(self):
        lines = []
        for value, sign, exponent, significand, denormal, result in zip(
            self.value.flatten(), self.sign.flatten(), self.exponent.flatten(),
            self.significand.flatten(), self.is_denormal.flatten(), self.rounding_value.flatten(),
        ):
            lines.append(
                f"value: {value.item()}, sign: {sign.item()}, exp: {exponent.item()}, "
                f"sig: {significand.item():0{self.sig_bits}b}, "
                f"emulated value: {result.item()}, denorm: {denormal.item()}"
            )
        return "\n".join(lines)
