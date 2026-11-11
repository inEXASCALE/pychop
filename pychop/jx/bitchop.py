"""JAX bit-level floating-point chopping backend."""

import jax
import jax.numpy as jnp
from jax import device_put, random
from .._jax_compat import enable_x64

from .._bitchop_common import (
    normalize_rmode, overflow_to_infinity, validate_format, validate_input_range,
)


class Bitchop:
    """Quantize float32 or float64 JAX arrays into a binary format."""

    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, random_state=42,
                 device="cpu", allow_wider_source_range=False):
        validate_format(exp_bits, sig_bits)
        available_devices = jax.devices()
        if device is None:
            target_device = None
        elif hasattr(device, "platform"):
            if device not in available_devices:
                raise ValueError(f"Unavailable JAX device: {device}")
            target_device = device
        elif isinstance(device, str):
            platform, _, index = device.lower().partition(":")
            matches = [d for d in available_devices if d.platform == platform]
            if not matches or (index and (not index.isdigit() or int(index) >= len(matches))):
                raise ValueError(
                    f"Invalid or unavailable device: {device}. "
                    f"Available: {[d.platform for d in available_devices]}"
                )
            target_device = matches[int(index)] if index else matches[0]
        else:
            raise ValueError(f"Invalid JAX device: {device}")

        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = normalize_rmode(rmode)
        self.subnormal = subnormal
        self.allow_wider_source_range = allow_wider_source_range
        self.device = target_device
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = 1 - self.bias
        self.mask_sig = (1 << sig_bits) - 1
        self.rng_key = random.PRNGKey(random_state)

    def __call__(self, values):
        # Keep float32 tracing in 32-bit arithmetic. A context-local setting
        # supplies 64-bit operations only for float64 inputs.
        input_dtype = getattr(values, "dtype", None)
        if input_dtype is None or str(input_dtype) == "float64":
            with enable_x64(True):
                return self._call_with_x64(values)
        return self._call_with_x64(values)

    def _call_with_x64(self, values):
        self.value = jnp.asarray(values)
        if self.device is not None:
            self.value = device_put(self.value, self.device)
        self.dtype = self.value.dtype
        if self.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("Input must be jnp.float32 or jnp.float64")
        if not self.allow_wider_source_range:
            validate_input_range(self.exp_bits, 8 if self.dtype == jnp.float32 else 11)
        if self.value.ndim == 0:
            self.value = self.value[None]
        self.sign, self.exponent, self.significand, self.is_denormal, self.rounding_value = self._convert()
        return self.rounding_value

    def _convert(self):
        value = self.value
        if self.dtype == jnp.float32:
            bits = value.view(jnp.int32)
            source_sig_bits, source_bias, source_min_exp = 23, 127, -126
            raw_exp = (bits >> 23) & 0xff
        else:
            bits = value.view(jnp.int64)
            source_sig_bits, source_bias, source_min_exp = 52, 1023, -1022
            raw_exp = (bits >> 52) & 0x7ff

        sign = (bits >> (source_sig_bits + (8 if self.dtype == jnp.float32 else 11))) & 1
        fraction = bits & ((1 << source_sig_bits) - 1)
        special = raw_exp == (2 * source_bias + 1)
        zero = (raw_exp == 0) & (fraction == 0)
        source_power = jnp.where(raw_exp == 0, source_min_exp, raw_exp - source_bias) - source_sig_bits
        whole_sig = jnp.where(raw_exp == 0, fraction, fraction | (1 << source_sig_bits))
        # JAX's float32 frexp flushes subnormal inputs on some platforms.
        _, fraction_exp = jnp.frexp(jnp.maximum(fraction, 1).astype(value.dtype))
        actual_exp = jnp.where(
            raw_exp == 0,
            fraction_exp.astype(bits.dtype) - 1 + source_min_exp - source_sig_bits,
            raw_exp - source_bias,
        )
        scale_exp = jnp.maximum(actual_exp, self.min_exp)
        shift = scale_exp - self.sig_bits - source_power

        right_shift = jnp.clip(shift, 0, 63 if self.dtype == jnp.float64 else 31)
        left_shift = jnp.clip(-shift, 0, 52)
        right_quotient = whole_sig >> right_shift
        quotient = jnp.where(shift > 0, right_quotient, whole_sig << left_shift)
        remainder = jnp.where(shift > 0, whole_sig - (right_quotient << right_shift), 0)
        inexact = remainder != 0
        halfway = jnp.ones_like(remainder) << jnp.clip(shift - 1, 0, 62 if self.dtype == jnp.float64 else 30)

        if self.rmode in (0, 1):
            odd = (quotient & 1) != 0
            tie_up = odd if self.rmode == 1 else ~odd
            round_up = inexact & ((remainder > halfway) | ((remainder == halfway) & tie_up))
        elif self.rmode in (7, 8):
            round_up = inexact & (remainder > halfway)
            if self.rmode == 8:
                round_up = round_up | (inexact & (remainder == halfway))
        elif self.rmode == 9:
            round_up = inexact & ((quotient & 1) == 0)
        elif self.rmode == 2:
            round_up = inexact & (sign == 0)
        elif self.rmode == 3:
            round_up = inexact & (sign == 1)
        elif self.rmode == 4:
            round_up = jnp.zeros_like(inexact)
        else:
            self.rng_key, subkey = random.split(self.rng_key)
            random_value = random.uniform(subkey, shape=value.shape, dtype=value.dtype)
            if self.rmode == 5:
                probability = jnp.ldexp(remainder.astype(value.dtype), -shift.astype(jnp.int32))
                round_up = inexact & (random_value < probability)
            else:
                round_up = inexact & (random_value < 0.5)

        rounded = quotient + round_up.astype(bits.dtype)
        carry = rounded >= (1 << (self.sig_bits + 1))
        rounded = jnp.where(carry, rounded >> 1, rounded)
        scale_exp = scale_exp + carry.astype(bits.dtype)

        overflow = scale_exp > self.max_exp
        flushed = (rounded < (1 << self.sig_bits)) & (not self.subnormal) & ~overflow & ~special
        is_denormal = (rounded > 0) & (rounded < (1 << self.sig_bits)) & ~flushed & ~overflow & ~special
        zero_result = zero | flushed | ((rounded == 0) & ~overflow & ~special)
        biased_exp = jnp.where(is_denormal | zero_result, 0, scale_exp + self.bias)
        sig_int = jnp.where(is_denormal, rounded, rounded - (1 << self.sig_bits))

        magnitude = jnp.ldexp(rounded.astype(value.dtype), (scale_exp - self.sig_bits).astype(jnp.int32))
        finite = jnp.copysign(magnitude, value)
        if overflow_to_infinity(self.rmode, False) and overflow_to_infinity(self.rmode, True):
            overflow_value = jnp.copysign(jnp.full_like(value, jnp.inf), value)
        else:
            max_value = jnp.ldexp(
                jnp.full_like(value, 2.0 - 2.0 ** -self.sig_bits),
                jnp.full_like(scale_exp, self.max_exp, dtype=jnp.int32),
            )
            infinity = jnp.full_like(value, jnp.inf)
            to_inf = (sign == 0) if self.rmode == 2 else (sign == 1) if self.rmode == 3 else jnp.zeros_like(zero)
            overflow_value = jnp.copysign(jnp.where(to_inf, infinity, max_value), value)
        result = jnp.where(overflow, overflow_value, finite)
        result = jnp.where(zero_result, jnp.copysign(jnp.zeros_like(value), value), result)
        result = jnp.where(special, value, result)

        # JAX's floating ldexp/cast also flushes source-format subnormals. Build
        # those exact results from their source IEEE bit pattern instead.
        _, rounded_exp = jnp.frexp(jnp.maximum(rounded, 1).astype(value.dtype))
        output_exp = rounded_exp.astype(bits.dtype) - 1 + scale_exp - self.sig_bits
        source_subnormal = (rounded > 0) & (output_exp < source_min_exp) & ~overflow & ~flushed & ~special
        bit_shift = scale_exp - self.sig_bits - (source_min_exp - source_sig_bits)
        source_fraction = jnp.where(
            bit_shift >= 0,
            rounded << jnp.clip(bit_shift, 0, 63),
            rounded >> jnp.clip(-bit_shift, 0, 63),
        )
        if self.dtype == jnp.float32:
            subnormal_bits = ((sign << 31) | source_fraction).astype(jnp.int32)
            subnormal_value = subnormal_bits.view(jnp.float32)
        else:
            subnormal_bits = (sign << 63) | source_fraction
            subnormal_value = subnormal_bits.view(jnp.float64)
        result = jnp.where(source_subnormal, subnormal_value, result)

        special_exp = (1 << self.exp_bits) - 1
        overflow_inf = overflow & jnp.isinf(result)
        biased_exp = jnp.where(overflow_inf, special_exp, biased_exp)
        biased_exp = jnp.where(overflow & ~overflow_inf, self.max_exp + self.bias, biased_exp)
        biased_exp = jnp.where(special, special_exp, biased_exp)
        sig_int = jnp.where(zero_result | overflow_inf, 0, sig_int)
        sig_int = jnp.where(overflow & ~overflow_inf, self.mask_sig, sig_int)
        sig_int = jnp.where(special & jnp.isnan(value), 1, sig_int)
        sig_int = jnp.where(special & jnp.isinf(value), 0, sig_int)
        return sign.astype(jnp.uint8), biased_exp.astype(bits.dtype), sig_int.astype(bits.dtype), is_denormal, result

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
        return f"Device: {self.device.device_kind}\n" + "\n".join(lines)
