"""Lazy native tensor kernels for the public P3109 API."""


def backend_of(value):
    """Recognize arrays, tracers and framework subclasses without importing them."""
    for cls in type(value).__mro__:
        module = cls.__module__
        if module.startswith("torch"):
            return "torch"
        if module.startswith("tensorflow"):
            return "tensorflow"
        if module.startswith(("jax.", "jaxlib.")):
            return "jax"
    return None


def adapter(name):
    if name == "torch":
        from .torch import TorchOps
        return TorchOps()
    if name == "tensorflow":
        from .tensorflow import TensorFlowOps
        return TensorFlowOps()
    from .jax import JaxOps
    return JaxOps()
