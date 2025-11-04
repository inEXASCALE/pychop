import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

class LightChop:
    """
    A class to simulate different floating-point precisions and rounding modes
    for JAX arrays. This code implements a custom floating-point precision simulator
    that mimics IEEE 754 floating-point representation with configurable exponent bits (exp_bits),
    significand bits (sig_bits), and various rounding modes (rmode). 
    It uses JAX arrays for efficient computation and handles special cases like zeros,
    infinities, NaNs, and subnormal numbers. The code follows IEEE 754 conventions for sign, 
    exponent bias, implicit leading 1 (for normal numbers), and subnormal number handling.

    Parameters
    ----------
    exp_bits: int 
        Number of bits for exponent.
    sig_bits : int
        Number of bits for significand (significant digits)
    rmode : int
        Rounding mode to use when quantizing the significand. Options are:
        - 1 : Round to nearest value, ties to even (IEEE 754 default).
        - 2 : Round towards plus infinity (round up).
        - 3 : Round towards minus infinity (round down).
        - 4 : Truncate toward zero (no rounding up).
        - 5 : Stochastic rounding proportional to the fractional part.
        - 6 : Stochastic rounding with 50% probability.
        - 7 : Round to nearest value, ties to zero.
        - 8 : Round to nearest value, ties to away.
        - 9 : Round to odd.
    random_state : int, default=42
        Random seed for stochastic rounding.
    subnormal : bool, default=True
        Whether to support subnormal numbers.
    chunk_size : int, default=1000
        Chunk size for processing large arrays (not used in this implementation).
    """
    
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True, 
                 chunk_size: int = 1000, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.subnormal = subnormal
        self.sig_steps = 2 ** sig_bits
        self.rng_key = jax.random.PRNGKey(random_state)

        self._to_custom_float_vmap = vmap(self._to_custom_float, in_axes=0)
        self._quantize_jit = jit(self._quantize_components, static_argnums=(7,))
        self._quantize_vmap = vmap(self._quantize_jit, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0))

    def _to_custom_float(self, x):
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = jnp.isinf(x)
        nan_mask = jnp.isnan(x)
        
        exponent = jnp.floor(jnp.log2(jnp.maximum(abs_x, 1e-38)))
        significand = abs_x / (2.0 ** exponent)
        
        if self.subnormal:
            subnormal_mask = (exponent < self.min_exp)
            significand = jnp.where(subnormal_mask, abs_x / (2.0 ** self.min_exp), significand)
            exponent = jnp.where(subnormal_mask, self.min_exp, exponent)
        else:
            subnormal_mask = (exponent < self.min_exp)
            significand = jnp.where(subnormal_mask, 0.0, significand)
            exponent = jnp.where(subnormal_mask, 0, exponent)
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask

    def _quantize_components(self, x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, rmode, key):
        exp_max = 2 ** self.exp_bits - 1
        exponent = jnp.clip(exponent, 0, exp_max)
        
        normal_mask = (exponent > 0) & (exponent < exp_max)
        subnormal_mask = (exponent == 0) & (significand > 0) if self.subnormal else jnp.zeros_like(x, dtype=bool)
        sig_normal = significand - 1.0
        
        sig_steps = self.sig_steps
        sig_scaled = sig_normal * sig_steps
        sig_sub_scaled = significand * sig_steps if self.subnormal else sig_scaled
        
        def nearest(sig_scaled, sig_sub_scaled, subnormal_mask):
            sig_q = jnp.round(sig_scaled) / sig_steps
            return jnp.where(subnormal_mask, jnp.round(sig_sub_scaled) / sig_steps, sig_q) if self.subnormal else sig_q

        def plus_inf(sig_scaled, sig_sub_scaled, subnormal_mask, sign):
            sig_q = jnp.where(sign > 0, jnp.ceil(sig_scaled), jnp.floor(sig_scaled)) / sig_steps
            if self.subnormal:
                sig_q_sub = jnp.where(sign > 0, jnp.ceil(sig_sub_scaled), jnp.floor(sig_sub_scaled)) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        def minus_inf(sig_scaled, sig_sub_scaled, subnormal_mask, sign):
            sig_q = jnp.where(sign > 0, jnp.floor(sig_scaled), jnp.ceil(sig_scaled)) / sig_steps
            if self.subnormal:
                sig_q_sub = jnp.where(sign > 0, jnp.floor(sig_sub_scaled), jnp.ceil(sig_sub_scaled)) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        def towards_zero(sig_scaled, sig_sub_scaled, subnormal_mask):
            sig_q = jnp.floor(sig_scaled) / sig_steps
            return jnp.where(subnormal_mask, jnp.floor(sig_sub_scaled) / sig_steps, sig_q) if self.subnormal else sig_q

        def stoc_prop(sig_scaled, sig_sub_scaled, subnormal_mask, key):
            floor_val = jnp.floor(sig_scaled)
            fraction = sig_scaled - floor_val
            prob = jax.random.uniform(key, shape=())
            sig_q = jnp.where(prob < fraction, floor_val + 1, floor_val) / sig_steps
            if self.subnormal:
                sub_floor = jnp.floor(sig_sub_scaled)
                sub_fraction = sig_sub_scaled - sub_floor
                sig_q_sub = jnp.where(prob < sub_fraction, sub_floor + 1, sub_floor) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        def stoc_equal(sig_scaled, sig_sub_scaled, subnormal_mask, key):
            floor_val = jnp.floor(sig_scaled)
            prob = jax.random.uniform(key, shape=())
            sig_q = jnp.where(prob < 0.5, floor_val, floor_val + 1) / sig_steps
            if self.subnormal:
                sub_floor = jnp.floor(sig_sub_scaled)
                sig_q_sub = jnp.where(prob < 0.5, sub_floor, sub_floor + 1) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        def nearest_ties_zero(sig_scaled, sig_sub_scaled, subnormal_mask, sign):
            floor_val = jnp.floor(sig_scaled)
            is_half = jnp.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = jnp.where(is_half, jnp.where(sign >= 0, floor_val, floor_val + 1), jnp.round(sig_scaled)) / sig_steps
            if self.subnormal:
                sub_floor = jnp.floor(sig_sub_scaled)
                sub_is_half = jnp.abs(sig_sub_scaled - sub_floor - 0.5) < 1e-6
                sig_q_sub = jnp.where(sub_is_half, jnp.where(sign >= 0, sub_floor, sub_floor + 1), 
                                    jnp.round(sig_sub_scaled)) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        def nearest_ties_away(sig_scaled, sig_sub_scaled, subnormal_mask, sign):
            floor_val = jnp.floor(sig_scaled)
            is_half = jnp.abs(sig_scaled - floor_val - 0.5) < 1e-6
            sig_q = jnp.where(is_half, jnp.where(sign >= 0, floor_val + 1, floor_val), jnp.round(sig_scaled)) / sig_steps
            if self.subnormal:
                sub_floor = jnp.floor(sig_sub_scaled)
                sub_is_half = jnp.abs(sig_sub_scaled - sub_floor - 0.5) < 1e-6
                sig_q_sub = jnp.where(sub_is_half, jnp.where(sign >= 0, sub_floor + 1, sub_floor), 
                                    jnp.round(sig_sub_scaled)) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        def round_to_odd(sig_scaled, sig_sub_scaled, subnormal_mask):
            rounded = jnp.round(sig_scaled)
            sig_q = jnp.where(rounded % 2 == 0, 
                            rounded + jnp.where(sig_scaled >= rounded, 1, -1), 
                            rounded) / sig_steps
            if self.subnormal:
                sub_rounded = jnp.round(sig_sub_scaled)
                sig_q_sub = jnp.where(sub_rounded % 2 == 0,
                                    sub_rounded + jnp.where(sig_sub_scaled >= sub_rounded, 1, -1),
                                    sub_rounded) / sig_steps
                return jnp.where(subnormal_mask, sig_q_sub, sig_q)
            return sig_q

        rounding_fns = {
            1: nearest,
            2: lambda s, ss, sm: plus_inf(s, ss, sm, sign),
            3: lambda s, ss, sm: minus_inf(s, ss, sm, sign),
            4: towards_zero,
            5: lambda s, ss, sm: stoc_prop(s, ss, sm, key),
            6: lambda s, ss, sm: stoc_equal(s, ss, sm, key),
            7: lambda s, ss, sm: nearest_ties_zero(s, ss, sm, sign),
            8: lambda s, ss, sm: nearest_ties_away(s, ss, sm, sign),
            9: round_to_odd,
        }
        
        if rmode not in rounding_fns:
            raise ValueError(f"Unsupported rounding mode: {rmode}")
        
        sig_q = rounding_fns[rmode](sig_scaled, sig_sub_scaled, subnormal_mask)
        
        result = jnp.where(normal_mask, sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias)), 0.0)
        if self.subnormal:
            result = jnp.where(subnormal_mask, sign * sig_q * (2.0 ** self.min_exp), result)
        result = jnp.where(zero_mask, 0.0, result)
        result = jnp.where(inf_mask, sign * jnp.inf, result)
        result = jnp.where(nan_mask, jnp.nan, result)
        
        return result

    def quantize(self, x):
        x = jnp.asarray(x)
        if x.ndim == 0:  # Handle scalar input
            sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
            key = jax.random.split(self.rng_key, 2)[1]
            self.rng_key = jax.random.split(self.rng_key, 2)[0]  # Update rng_key
            return self._quantize_jit(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode, key)
        else:  # Handle array input
            keys = jax.random.split(self.rng_key, x.size + 1)
            self.rng_key = keys[0]  # Update the key for the next call
            sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float_vmap(x)
            return self._quantize_vmap(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode, keys[1:])

    def __call__(self, x):
        return self.quantize(x)