# Backend Import and Dispatch

This document explains how pychop imports optional backend packages and selects
a backend for each input.

## Import Behavior

`import pychop` is intentionally lightweight.

A bare package import should load NumPy, but it should not import optional heavy
frameworks:

- TensorFlow
- PyTorch
- JAX or jaxlib
- pandas

Those packages are imported only when a user explicitly asks for functionality
that requires them.

This avoids TensorFlow CUDA/XLA plugin registration warnings during
`import pychop`, and it keeps startup time low for users who only need NumPy or
format metadata.

## Backend Selection

pychop uses the `chop_backend` environment variable, controlled through
`pychop.backend(...)`.

Supported values are:

- `"auto"`
- `"numpy"`
- `"torch"`
- `"jax"`
- `"tensorflow"` or `"tf"`

The default is `"auto"`.

### Explicit Backend

When the backend is fixed explicitly, pychop routes operations to that backend:

```python
import pychop

pychop.backend("torch")
```

The backend package is imported on first actual use of backend-specific
functionality. After that, Python's normal `sys.modules` cache prevents repeated
framework imports.

### Auto Backend

In `"auto"` mode, pychop detects the backend from the input object type.

For example:

- `numpy.ndarray` -> NumPy backend
- `torch.Tensor` -> PyTorch backend
- JAX arrays -> JAX backend
- TensorFlow tensors or variables -> TensorFlow backend

The type detection itself is lightweight. It checks the object's concrete type
module name and does not import TensorFlow, PyTorch, or JAX just to perform the
check.

## Conversion Behavior

pychop avoids unnecessary conversion when the input already belongs to the
selected backend:

- A NumPy array routed to NumPy is used directly.
- A PyTorch tensor routed to PyTorch is returned directly by `to_torch_tensor`.
- A JAX array routed to JAX is returned directly by `to_jax_array`.
- A TensorFlow tensor routed to TensorFlow is returned directly by
  `to_tensorflow_tensor`.

Data copies happen only when the input must cross backend boundaries, for
example converting a PyTorch tensor to NumPy or a JAX array to TensorFlow.
Those conversions are explicit and may emit warnings when device-to-host or
cross-framework copies are involved.

## Implementation Caching

Backend-specific implementation objects are cached differently across front-end
classes.

### Cached by Backend

`Chopi` and `Chopf` keep a backend implementation once it has been resolved.
They reuse that implementation while the resolved backend stays the same.

### Reuse in `Chop`

In `"auto"` mode, `Chop` detects the input backend on each call and reuses its
current backend implementation while that backend stays the same. Switching to
a different backend creates a new implementation and restarts its random-number
sequence from the configured seed.

## Recommended Usage

For predictable performance in tight loops, set the backend explicitly when the
input framework is known:

```python
import pychop

pychop.backend("numpy")
# or
pychop.backend("torch")
```

Use `"auto"` when code needs to accept arrays from multiple frameworks and the
small per-call type check is acceptable.

