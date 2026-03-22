import jax
import jax.numpy as jnp
from jax import random, device_put




class Bitchop(object):
    """
    Parameters
    ----------
    exp_bits : int
        Number of bits for the exponent in the target format. Determines the range
        of representable values (e.g., 5 bits gives a bias of 15, range -14 to 15).

    sig_bits : int
        Number of bits for the significand (mantissa) in the target format, excluding
        the implicit leading 1 for normalized numbers (e.g., 4 bits allows 0 to 15 plus implicit 1).

    subnormal : boolean
        Whether or not support subnormal numbers are supported.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int, default=1
        Rounding mode to use when quantizing the significand. Options are:
        - 1: Round to nearest value, ties to even (IEEE 754 default).
        - 0: Round to nearest value, ties to odd.
        - 2: Round towards plus infinity (round up).
        - 3: Round towards minus infinity (round down).
        - 4: Truncate toward zero (no rounding up).
        - 5: Stochastic rounding proportional to the fractional part.
        - 6: Stochastic rounding with 50% probability.
        
    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    device : str or torch.device, optional, default="cpu" 
        Device to perform computations on (e.g., "cpu", "cuda").

    subnormal (bool, optional): If True, supports denormalized numbers (subnormals) when
        the exponent underflows, shifting the significand. If False, underflows result in zero.
        Defaults to True.

    Methods
    ----------
    bitchop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """

    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, random_state=42, device="cpu"):
        # Validate and set device
        # Enable 64-bit precision (optional, based on your earlier preference)
        jax.config.update("jax_enable_x64", True)
        available_devices = jax.devices()
        device_map = {d.device_kind.lower(): d for d in available_devices}
        
        # Handle device specification (e.g., "cpu", "gpu", "gpu:0")
        if device.lower() == "gpu" and "gpu" in device_map:
            target_device = device_map["gpu"]
        elif device.lower() == "cpu" and "cpu" in device_map:
            target_device = device_map["cpu"]
        elif device.lower().startswith("gpu:") and "gpu" in device_map:
            gpu_idx = int(device.split(":")[1])
            target_device = available_devices[gpu_idx] if gpu_idx < len(available_devices) else device_map["gpu"]
        else:
            raise ValueError(f"Invalid or unavailable device: {device}. Available: {[d.device_kind for d in available_devices]}")
        
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits

        self.rmode = rmode
        
        self.subnormal = subnormal
        self.device = target_device
        
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = -self.bias + 1
        self.mask_sig = (1 << sig_bits) - 1
        self.rng_key = random.PRNGKey(random_state)  # JAX PRNG key, placed implicitly on device


    def __call__(self, values):

        # Convert input to JAX array and place it on the specified device
        self.value = device_put(jnp.asarray(values), self.device)
        self.dtype = self.value.dtype
        if self.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("Input must be jnp.float32 or jnp.float64")
        
        # Ensure value is at least 1D
        if self.value.ndim == 0:
            self.value = self.value[None]
        
        # Initialize output arrays on the specified device
        self.sign = device_put(jnp.zeros_like(self.value, dtype=jnp.uint8), self.device)
        self.exponent = device_put(jnp.zeros_like(self.value, dtype=jnp.int32), self.device)
        self.significand = device_put(jnp.zeros_like(self.value, dtype=jnp.int32), self.device)
        self.is_denormal = device_put(jnp.zeros_like(self.value, dtype=jnp.bool_), self.device)
        self.rounding_value = device_put(jnp.zeros_like(self.value, dtype=self.dtype), self.device)
        
        # Perform conversion
        self.sign, self.exponent, self.significand, self.is_denormal, self.rounding_value = self._convert()
        return self.rounding_value

    def _extract_components(self):
        if self.dtype == jnp.float32:
            bits = self.value.view(jnp.int32)
            sign = (bits >> 31) & 1
            exp = ((bits >> 23) & 0xFF) - 127
            mantissa = bits & ((1 << 23) - 1)
            mantissa_bits = 23
            min_exp = -126
            bias = 127
        else:  # jnp.float64
            bits = self.value.view(jnp.int64)
            sign = (bits >> 63) & 1
            exp = ((bits >> 52) & 0x7FF) - 1023
            mantissa = bits & ((1 << 52) - 1)
            mantissa_bits = 52
            min_exp = -1022
            bias = 1023
        
        is_zero = (exp == -bias) & (mantissa == 0)
        is_denorm = (exp == -bias) & (mantissa != 0)
        
        mantissa_norm = jnp.where(
            is_denorm,
            mantissa.astype(jnp.float64) / (1 << mantissa_bits),
            1.0 + mantissa.astype(jnp.float64) / (1 << mantissa_bits)
        ).astype(self.dtype)
        
        exp = jnp.where(is_denorm, jnp.array(min_exp, dtype=jnp.int32), exp)
        
        return sign.astype(jnp.uint8), exp.astype(jnp.int32), mantissa_norm, is_zero

    def _adjust_to_format(self, sign, exp, mantissa):
        mantissa_bits = (mantissa * (1 << (self.sig_bits + 1))).astype(jnp.int64) & ((1 << (self.sig_bits + 1)) - 1)
        exact_mantissa = mantissa_bits >> 1
        remainder = mantissa_bits & 1
        half_bit = (remainder << 1) & (mantissa_bits & 2)

        if self.rmode == 1:
            round_up = (remainder != 0) & (half_bit | (exact_mantissa & 1))
            rounded = exact_mantissa + round_up.astype(jnp.int64)
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 0:
            round_up = (remainder != 0) & (half_bit & ~(exact_mantissa & 1))
            rounded = exact_mantissa + round_up.astype(jnp.int64)
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 2:  # Round up
            round_up = (remainder != 0) & (sign == 0)  # Positive numbers round up
            rounded = exact_mantissa + round_up.astype(jnp.int64)
            did_round_up = rounded > exact_mantissa
            
        elif self.rmode == 3:  # Round down
            round_up = (remainder != 0) & (sign == 1)  # Negative numbers round down
            rounded = exact_mantissa + round_up.astype(jnp.int64)
            did_round_up = rounded > exact_mantissa
        
        elif self.rmode == 4:
            rounded = exact_mantissa
            did_round_up = jnp.zeros_like(rounded, dtype=jnp.bool_)

        elif self.rmode == 5:
            prob = (mantissa * (1 << self.sig_bits) - exact_mantissa.astype(self.dtype))
            # Generate random values with JAX RNG
            key, subkey = random.split(self.rng)
            rand = random.uniform(subkey, shape=exact_mantissa.shape, dtype=self.dtype)
            rounded = exact_mantissa + (rand < prob).astype(jnp.int64)
            did_round_up = rounded > exact_mantissa
            self.rng = key  # Update RNG state

        elif self.rmode == 6:
            key, subkey = random.split(self.rng)
            rand = random.uniform(subkey, shape=exact_mantissa.shape, dtype=self.dtype)
            rounded = exact_mantissa + (rand < 0.5).astype(jnp.int64)
            did_round_up = rounded > exact_mantissa
            self.rng = key  # Update RNG state
        else:
            raise ValueError("Unknown rmode mode")

        overflow = did_round_up & (rounded >= (1 << self.sig_bits))
        rounded = jnp.where(overflow, rounded >> 1, rounded)
        exp = exp + overflow.astype(jnp.int32)

        overflow_mask = exp > self.max_exp
        underflow_mask = exp < self.min_exp
        
        if overflow_mask.any():
            raise OverflowError(f"Exponent too large in {overflow_mask.sum()} elements")
        
        if underflow_mask.any():
            if not self.subnormal:
                sign = jnp.where(underflow_mask, jnp.array(0, dtype=jnp.uint8), sign)
                exp = jnp.where(underflow_mask, jnp.array(0, dtype=jnp.int32), exp)
                rounded = jnp.where(underflow_mask, jnp.array(0, dtype=jnp.int32), rounded)
                is_denormal = jnp.zeros_like(underflow_mask)
            else:
                shift = jnp.clip(self.min_exp - exp, min=0)
                rounded = jnp.where(underflow_mask, rounded >> shift, rounded)
                exp = jnp.where(underflow_mask, jnp.array(self.min_exp, dtype=jnp.int32), exp)
                is_denormal = underflow_mask
        else:
            is_denormal = jnp.zeros_like(exp, dtype=jnp.bool_)

        biased_exp = jnp.where(is_denormal, jnp.array(0, dtype=jnp.int32), exp + self.bias)
        sig_int = rounded & self.mask_sig
        reconstructed = self._reconstruct(sign, biased_exp, sig_int, is_denormal)
        return sign, biased_exp, sig_int, is_denormal, reconstructed

    def _reconstruct(self, sign, exponent, significand, is_denormal):
        zero_mask = (exponent == 0) & (significand == 0)
        sig_value = jnp.where(
            is_denormal,
            significand.astype(self.dtype) / (1 << self.sig_bits),
            1.0 + significand.astype(self.dtype) / (1 << self.sig_bits)
        )
        exp_value = jnp.where(
            is_denormal,
            jnp.array(self.min_exp, dtype=jnp.int32),
            exponent - self.bias
        )
        return jnp.where(
            zero_mask,
            jnp.array(0.0, dtype=self.dtype),
            ((-1) ** sign.astype(self.dtype)) * sig_value * (2.0 ** exp_value.astype(self.dtype))
        ).astype(self.dtype)

    def _convert(self):
        sign, exp, mantissa, is_zero = self._extract_components()
        return self._adjust_to_format(sign, exp, mantissa)

    def __str__(self):
        lines = []
        for i in range(self.value.size):
            val = self.value.flatten()[i].item()
            s = self.sign.flatten()[i].item()
            e = self.exponent.flatten()[i].item()
            sig = bin(self.significand.flatten()[i].item())[2:].zfill(self.sig_bits)
            recon = self.rounding_value.flatten()[i].item()
            denorm = self.is_denormal.flatten()[i].item()
            lines.append(f"value: {val}, sign: {s}, exp: {e}, sig: {sig}, emulated value: {recon}, denorm: {denorm}")
        return f"Device: {self.device.device_kind}\n" + "\n".join(lines)

# Example usage
if __name__ == "__main__":
    # Test with float32 input on CPU
    values_float32 = jnp.array([3.14159, 0.1, -2.718], dtype=jnp.float32)
    bf_float32_cpu = bitchop(exp_bits=5, sig_bits=4, rmode=1, device="cpu")
    emulated_values = bf_float32_cpu(values_float32)
    print("Float32 emulated input(CPU):", emulated_values)
    print()

    # Test with float64 input on GPU (if available)
    # values_float64 = jnp.array([3.14159, 0.1, -2.718], dtype=jnp.float64)
    # bf_float64_gpu = BinaryFloat4(values_float64, exp_bits=5, sig_bits=4, rmode="nearest_even", device="gpu")
    # print("Float64 Input (GPU):")
    # print(bf_float64_gpu)