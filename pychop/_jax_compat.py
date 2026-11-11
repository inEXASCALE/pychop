"""Compatibility helpers for JAX APIs that moved between releases."""


def enable_x64(value):
    """Return a scoped x64 setting on both old and current JAX releases."""
    import jax

    context = getattr(jax, "enable_x64", None)
    if context is None:
        context = jax.experimental.enable_x64
    return context(value)
