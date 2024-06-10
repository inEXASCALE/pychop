import jax.numpy as jnp

def round_clamp(x, bits=8):
    x = jnp.clip(x, a_min=0, a_max=2**(bits)-1)
    x = jnp.round(x * 2**(bits)) / (2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    x_f = jnp.sign(x)*round_clamp(jnp.abs(x) - jnp.floor(jnp.abs(x)), fbits)
    x_i = jnp.sign(x)*round_clamp(jnp.floor(jnp.abs(x)), ibits)
    return (x_i + x_f)