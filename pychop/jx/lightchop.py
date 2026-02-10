import jax
import jax.numpy as jnp
from jax import jit

class LightChop_:
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
        # Precompute constants
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.subnormal = subnormal
        self.sig_steps = 2.0 ** sig_bits
        self.chunk_size = chunk_size
        # Random state management
        self.rng_key = jax.random.PRNGKey(random_state)

        # JIT-compile the quantization core. 
        # static_argnums=(7,) corresponds to 'rmode' to allow specialized kernels.
        self._quantize_core = jit(self._quantize_impl, static_argnums=(7,))

    def _to_custom_float(self, x):
        """Decomposes x into sign, exponent, and significand (vectorized)."""
        sign = jnp.sign(x)
        # Fix sign of 0 to be positive for standard handling, though IEEE supports -0.
        # Here we just preserve the sign bit as extracted.
        
        abs_x = jnp.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = jnp.isinf(x)
        nan_mask = jnp.isnan(x)
        
        # Calculate exponent
        # Use maximum(..., 1e-38) to avoid log2(0)
        exponent = jnp.floor(jnp.log2(jnp.maximum(abs_x, 1e-45)))
        significand = abs_x / (2.0 ** exponent)
        
        if self.subnormal:
            # If exponent is below min_exp, treat as subnormal
            subnormal_mask = (exponent < self.min_exp)
            # Recalculate significand for subnormals: value / min_subnormal_scale
            significand = jnp.where(subnormal_mask, abs_x / (2.0 ** self.min_exp), significand)
            exponent = jnp.where(subnormal_mask, self.min_exp, exponent)
        else:
            # Flush subnormals to zero
            subnormal_mask = (exponent < self.min_exp)
            significand = jnp.where(subnormal_mask, 0.0, significand)
            exponent = jnp.where(subnormal_mask, 0.0, exponent) # Set to arbitrary valid exp
            zero_mask = zero_mask | subnormal_mask

        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask

    def _quantize_impl(self, x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, rmode, noise):
        """
        Core quantization logic. 
        'noise' is a pre-generated array of random values [0, 1] used for stochastic modes.
        """
        exp_max_val = 2 ** self.exp_bits - 1
        
        # Clip exponent to representable range
        exponent = jnp.clip(exponent, 0, exp_max_val)
        
        # Identify normal vs subnormal in the target format
        # Note: 'significand' coming in is in [1.0, 2.0) for normals (mostly)
        
        # We need to handle the specific target format logic
        normal_mask = (exponent > 0) & (exponent < exp_max_val)
        
        # For our extracted components, subnormal handling in target:
        # If we allowed subnormals in decomposition, they are already scaled relative to min_exp.
        # We just need to check if the *target* exponent is 0.
        subnormal_mask = (exponent == 0) if self.subnormal else jnp.zeros_like(x, dtype=bool)

        # Scale significand to integer range for rounding
        # Normals: Remove implicit 1.0, scale fractional part
        # Subnormals: Scale full value
        sig_steps = self.sig_steps
        
        # Prepare values for rounding
        # For normals: we round (significand - 1.0) * steps
        # For subnormals: we round significand * steps
        sig_to_round_normal = (significand - 1.0) * sig_steps
        sig_to_round_sub = significand * sig_steps
        
        # Select the value to round based on whether it's normal or subnormal
        val_to_round = jnp.where(subnormal_mask, sig_to_round_sub, sig_to_round_normal)
        
        # --- Rounding Implementations ---
        
        def nearest(val):
            return jnp.round(val)

        def plus_inf(val, sgn):
            # If positive, ceil. If negative, floor.
            return jnp.where(sgn > 0, jnp.ceil(val), jnp.floor(val))

        def minus_inf(val, sgn):
            # If positive, floor. If negative, ceil.
            return jnp.where(sgn > 0, jnp.floor(val), jnp.ceil(val))

        def towards_zero(val):
            return jnp.floor(val)

        def stoc_prop(val, noise):
            # Stochastic proportional
            floor_val = jnp.floor(val)
            fraction = val - floor_val
            # Round up if random noise < fraction
            return jnp.where(noise < fraction, floor_val + 1.0, floor_val)

        def stoc_equal(val, noise):
            # Stochastic 50/50
            floor_val = jnp.floor(val)
            # Round up if random noise < 0.5
            return jnp.where(noise < 0.5, floor_val, floor_val + 1.0)

        def nearest_ties_zero(val, sgn):
            floor_val = jnp.floor(val)
            is_half = jnp.abs(val - floor_val - 0.5) < 1e-6
            # If half: round towards zero (floor for pos, ceil for neg? No, magnitude towards zero)
            # Floor is closer to zero for positive numbers. 
            # Actually "ties to zero" usually means round 1.5 -> 1, 2.5 -> 2? No that's ties to even.
            # Ties to zero: 1.5 -> 1, -1.5 -> -1.
            round_half = jnp.where(sgn >= 0, floor_val, floor_val + 1.0)
            return jnp.where(is_half, round_half, jnp.round(val))

        def nearest_ties_away(val, sgn):
            floor_val = jnp.floor(val)
            is_half = jnp.abs(val - floor_val - 0.5) < 1e-6
            # Ties away from zero: 1.5 -> 2, -1.5 -> -2
            round_half = jnp.where(sgn >= 0, floor_val + 1.0, floor_val)
            return jnp.where(is_half, round_half, jnp.round(val))
            
        def round_to_odd(val):
            rounded = jnp.round(val)
            # If result is even, nudge it to the nearest odd integer
            # We nudge towards the original value to minimize error
            # If val > rounded (e.g. 2.1 -> 2), nudge up (+1)
            # If val < rounded (e.g. 1.9 -> 2), nudge down (-1)
            diff = val - rounded
            nudge = jnp.sign(diff)
            # If diff is exactly 0 (integer), strictly speaking we force odd. 
            # Usually RTO is for sticky bits, here we just force odd integer.
            # Let's default nudge +1 if exactly even integer.
            nudge = jnp.where(nudge == 0, 1.0, nudge) 
            
            return jnp.where(rounded % 2 == 0, rounded + nudge, rounded)

        # Dispatcher
        if rmode == 1:
            rounded_val = nearest(val_to_round)
        elif rmode == 2:
            rounded_val = plus_inf(val_to_round, sign)
        elif rmode == 3:
            rounded_val = minus_inf(val_to_round, sign)
        elif rmode == 4:
            rounded_val = towards_zero(val_to_round)
        elif rmode == 5:
            rounded_val = stoc_prop(val_to_round, noise)
        elif rmode == 6:
            rounded_val = stoc_equal(val_to_round, noise)
        elif rmode == 7:
            rounded_val = nearest_ties_zero(val_to_round, sign)
        elif rmode == 8:
            rounded_val = nearest_ties_away(val_to_round, sign)
        elif rmode == 9:
            rounded_val = round_to_odd(val_to_round)
        else:
            # Fallback to nearest
            rounded_val = nearest(val_to_round)

        # Scale back down
        sig_q = rounded_val / sig_steps
        
        # Reconstruct
        # If normal: value = (1 + sig_q) * 2^(exp - bias)
        # If subnormal: value = sig_q * 2^(min_exp)
        
        res_normal = sign * (1.0 + sig_q) * (2.0 ** (exponent - self.bias))
        res_subnormal = sign * sig_q * (2.0 ** self.min_exp)
        
        result = jnp.where(subnormal_mask, res_subnormal, res_normal)
        
        # Apply standard masks
        result = jnp.where(zero_mask, 0.0, result)
        result = jnp.where(inf_mask, sign * jnp.inf, result)
        result = jnp.where(nan_mask, jnp.nan, result)
        
        return result

    def __call__(self, x):
        """
        Main entry point. Quantizes x.
        """
        x = jnp.asarray(x)
        
        # Split key only once per call
        key, subkey = jax.random.split(self.rng_key)
        self.rng_key = key
        
        # Generate noise tensor for stochastic modes (unused for deterministic modes, but cheap)
        # We generate it every time to keep the API clean, but JAX's lazy evaluation 
        # might optimize it out if rmode is static and deterministic.
        if self.rmode in [5, 6]:
            noise = jax.random.uniform(subkey, shape=x.shape)
        else:
            noise = jnp.zeros_like(x) # Dummy placeholder
            
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        
        return self._quantize_core(
            x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode, noise
        )