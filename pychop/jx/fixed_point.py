import jax.numpy as jnp

def round_clamp(x, bits=8):
    x = jnp.clip(x, a_min=0, a_max=2**(bits)-1)
    x = jnp.round(x * 2**(bits)) / (2**(bits))
    return x
    
def to_fixed_point(x, ibits=4, fbits=4):
    """
    Parameters
    ----------
    x : numpy.ndarray | jax.Array,
        The input array.    
    
    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 
        
    Methods
    ----------
    x_q : numpy.ndarray | jax.Array,
        The quantized array.
    """
    x_f = jnp.sign(x)*round_clamp(jnp.abs(x) - jnp.floor(jnp.abs(x)), fbits)
    x_i = jnp.sign(x)*round_clamp(jnp.floor(jnp.abs(x)), ibits)
    return (x_i + x_f)