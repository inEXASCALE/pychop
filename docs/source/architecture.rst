Architecture and numerical contracts
====================================

Public interfaces
-----------------

``pychop/chop.py``, ``integer.py`` and ``fixed_point.py`` are backend-dispatch
front ends. Backend kernels live under ``np/``, ``tch/``, ``jx/`` and ``tf/``.
``set_backend.py`` provides process-global selection; auto mode dispatches on
input type. ``Chop`` retains the most recently used backend implementation so
repeated calls avoid reconstruction and advance its random generator. Switching
backend replaces that implementation and restarts its state from the seed.
Avoid changing the global backend in concurrent application threads.

``builtin/`` provides scalar/array/tensor wrappers and precision transitions.
``layers.py`` and ``ptq.py`` dispatch model quantization to optional frameworks.
Layer factories have backend-specific coverage and constructors; identical names
do not imply identical framework signatures. JAX layer examples additionally
require Flax (and Optax for training).

``p3109.py`` is independent of global backend dispatch. An immutable
``P3109Format`` validates format invariants; ``P3109`` holds a rounding policy;
functions implement quantization, code encoding and decoding. Native tensor
operations live in ``_p3109/``: one shared numerical kernel and small lazy adapters
for Torch, TensorFlow and JAX. The tensor kernel uses exact IEEE bit decomposition
and native operations, supports graph compilation, and never rounds via NumPy.
Explicit STE gradients are implemented with framework custom-gradient hooks.
There is no
runtime gfloat dependency, mutable format cache or hidden stochastic generator.
The NumPy kernel uses exact ``frexp``/``ldexp`` operations and fewer temporary
arrays than the generic reference path. The NumPy path needs no JIT warm-up; tensor compilation is explicitly opt-in.

Rounding boundaries and storage
-------------------------------

A quantizer returns host floating-point values from a smaller representable set.
Adding two returned arrays uses host arithmetic until the next quantizer call.
Wrapping an operation rounds its output; it does not replace an internal BLAS
accumulator with low-precision hardware. Document these boundaries when reporting
experimental results, especially for matrix products and iterative solvers.

Integer code arrays are explicit storage representations. P3109 codes carry no
format metadata by themselves. Save the versioned quantizer policy beside the
codes.

Compatibility and change discipline
------------------------------------

Existing public quantizer names and backend selection remain available. The
P3109 API intentionally uses named rounding modes to avoid overloading the
historical ``Chop.rmode`` numbering. Draft evolution can require a new format
interpretation; pin the package revision and the reference draft in experiments.

Changes should preserve input ownership, validate format boundaries and include
regressions for numerical edge cases. Optional frameworks must remain lazy.
Tests should skip an unavailable optional backend at collection time, not fail
core-only installations. See :doc:`validation` for executable checks.
