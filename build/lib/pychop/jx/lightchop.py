import jax
import jax.numpy as jnp
from typing import Tuple
from jax.random import PRNGKey

class LightChop:
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, random_state: int = 42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        self.rmode = rmode
        self.key = PRNGKey(random_state)  # Initialize PRNG key with seed
        
    def _to_custom_float(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, 
                                                      jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = jnp.isinf(x)
        nan_mask = jnp.isnan(x)
        
        exponent = jnp.floor(jnp.log2(jnp.maximum(abs_x, 2.0**-24)))
        significand = abs_x / (2.0 ** exponent)
        
        subnormal_mask = (exponent < self.min_exp)
        significand = jnp.where(subnormal_mask, 
                              abs_x / (2.0 ** self.min_exp), 
                              significand)
        exponent = jnp.where(subnormal_mask, 
                           self.min_exp, 
                           exponent)
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask
    
    def _quantize_components(self, 
                           x: jnp.ndarray,
                           sign: jnp.ndarray, 
                           exponent: jnp.ndarray, 
                           significand: jnp.ndarray,
                           zero_mask: jnp.ndarray,
                           inf_mask: jnp.ndarray,
                           nan_mask: jnp.ndarray,
                           rmode: str) -> jnp.ndarray:
        
        exp_min = 0
        exp_max = 2**self.exp_bits - 1
        exponent = jnp.clip(exponent, exp_min, exp_max)
        
        significand_steps = 2 ** self.sig_bits
        normal_mask = (exponent > 0) & (exponent < exp_max)
        subnormal_mask = (exponent == 0)
        significand_normal = significand - 1.0
        
        if rmode in {"nearest", 1}:
            significand_q = jnp.round(significand_normal * significand_steps) / significand_steps
            significand_q = jnp.where(subnormal_mask,
                                    jnp.round(significand * significand_steps) / significand_steps,
                                    significand_q)
                
        elif rmode in {"plus_inf", 2}:
            significand_q = jnp.where(sign > 0,
                                    jnp.ceil(significand_normal * significand_steps),
                                    jnp.floor(significand_normal * significand_steps)) / significand_steps
            significand_q = jnp.where(subnormal_mask,
                                    jnp.where(sign > 0,
                                            jnp.ceil(significand * significand_steps),
                                            jnp.floor(significand * significand_steps)) / significand_steps,
                                    significand_q)
                
        elif rmode in {"minus_inf", 3}:
            significand_q = jnp.where(sign > 0,
                                    jnp.floor(significand_normal * significand_steps),
                                    jnp.ceil(significand_normal * significand_steps)) / significand_steps
            significand_q = jnp.where(subnormal_mask,
                                    jnp.where(sign > 0,
                                            jnp.floor(significand * significand_steps),
                                            jnp.ceil(significand * significand_steps)) / significand_steps,
                                    significand_q)
                
        elif rmode in {"towards_zero", 4}:
            significand_q = jnp.floor(significand_normal * significand_steps) / significand_steps
            significand_q = jnp.where(subnormal_mask,
                                    jnp.floor(significand * significand_steps) / significand_steps,
                                    significand_q)
                
        elif rmode in {"stoc_prop", 5}:
            significand_scaled = significand_normal * significand_steps
            floor_val = jnp.floor(significand_scaled)
            fraction = significand_scaled - floor_val
            # Split and advance the key for each random operation
            self.key, subkey = jax.random.split(self.key)
            prob = jax.random.uniform(subkey, shape=significand_scaled.shape)
            significand_q = jnp.where(prob < fraction, floor_val + 1, floor_val) / significand_steps
            significand_q = jnp.where(subnormal_mask,
                                    jnp.where(prob < (significand * significand_steps - jnp.floor(significand * significand_steps)),
                                            jnp.floor(significand * significand_steps) + 1,
                                            jnp.floor(significand * significand_steps)) / significand_steps,
                                    significand_q)
                
        elif rmode in {"stoc_equal", 6}:
            significand_scaled = significand_normal * significand_steps
            floor_val = jnp.floor(significand_scaled)
            # Split and advance the key for each random operation
            self.key, subkey = jax.random.split(self.key)
            prob = jax.random.uniform(subkey, shape=significand_scaled.shape)
            significand_q = jnp.where(prob < 0.5, floor_val, floor_val + 1) / significand_steps
            significand_q = jnp.where(subnormal_mask,
                                    jnp.where(prob < 0.5,
                                            jnp.floor(significand * significand_steps),
                                            jnp.floor(significand * significand_steps) + 1) / significand_steps,
                                    significand_q)
                
        elif rmode in {"nearest_ties_to_zero", 7}:
            significand_scaled = significand_normal * significand_steps
            floor_val = jnp.floor(significand_scaled)
            ceil_val = jnp.ceil(significand_scaled)
            is_half = jnp.abs(significand_scaled - floor_val - 0.5) < 1e-6
            significand_q = jnp.where(
                is_half,
                jnp.where(sign >= 0, floor_val, ceil_val),
                jnp.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = jnp.floor(significand_subnormal)
            sub_ceil = jnp.ceil(significand_subnormal)
            sub_is_half = jnp.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = jnp.where(
                subnormal_mask,
                jnp.where(
                    sub_is_half,
                    jnp.where(sign >= 0, sub_floor, sub_ceil),
                    jnp.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
            
        elif rmode in {"nearest_ties_to_away", 8}:
            significand_scaled = significand_normal * significand_steps
            floor_val = jnp.floor(significand_scaled)
            ceil_val = jnp.ceil(significand_scaled)
            is_half = jnp.abs(significand_scaled - floor_val - 0.5) < 1e-6
            significand_q = jnp.where(
                is_half,
                jnp.where(sign >= 0, ceil_val, floor_val),
                jnp.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = jnp.floor(significand_subnormal)
            sub_ceil = jnp.ceil(significand_subnormal)
            sub_is_half = jnp.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = jnp.where(
                subnormal_mask,
                jnp.where(
                    sub_is_half,
                    jnp.where(sign >= 0, sub_ceil, sub_floor),
                    jnp.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
    
        else:
            raise ValueError(f"Unsupported rounding mode: {rmode}")
        
        result = jnp.zeros_like(x)
        result = jnp.where(normal_mask,
                         sign * (1.0 + significand_q) * (2.0 ** (exponent - self.bias)),
                         result)
        result = jnp.where(subnormal_mask,
                         sign * significand_q * (2.0 ** self.min_exp),
                         result)
        result = jnp.where(zero_mask, 0.0, result)
        result = jnp.where(inf_mask, jnp.sign(x) * jnp.inf, result)
        result = jnp.where(nan_mask, jnp.nan, result)
        
        return result

    def quantize(self, x: jnp.ndarray) -> jnp.ndarray:
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask, self.rmode)

    def __call__(self, x: jnp.ndarray):
        return self.quantize(x)