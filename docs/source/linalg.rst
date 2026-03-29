Chopped linear algebra wrappers
======================================================

This page documents the high-level linear algebra helpers added under
``pychop.builtin.linalg``. These functions provide a consistent interface across
NumPy, JAX, and PyTorch backends while enforcing chopped-precision semantics at
the *function boundary*.

Overview
--------

Key features
^^^^^^^^^^^^

1. **Backend-dispatched linear algebra**
   The active backend is selected via ``pychop.backend(...)`` and dispatches to:

   - NumPy: ``numpy.linalg`` (and ``scipy.linalg`` for advanced routines)
   - JAX: ``jax.numpy.linalg`` / ``jax.scipy.linalg`` (when available)
   - Torch: ``torch.linalg`` (and legacy torch APIs as fallbacks)

2. **Chop + wrap at the call boundary**
   Inputs that are PyChop containers (e.g., ``CPArray``, ``CPJaxArray``,
   ``CPTensor``) are unwrapped to backend-native arrays/tensors, the backend
   routine is executed, then outputs are chopped and wrapped back.

3. **Scalar outputs as ``CPFloat``**
   Scalar-returning routines such as ``det``, ``norm``, and ``trace`` return a
   :class:`pychop.builtin.cpfloat.CPFloat` (instead of a raw Python float), so
   scalar arithmetic can continue to preserve chopped precision.

4. **SciPy-on-host fallback for advanced matrix functions**
   Some advanced routines (``logm``, ``sqrtm``, ``polar``) are not consistently
   available in JAX/Torch. For research workflows, an opt-in CPU fallback is
   provided:

   - ``allow_host_fallback=True`` transfers the data to CPU (NumPy), runs SciPy,
     then chops and wraps the result back into the current backend container.

   This is disabled by default because it breaks device placement and can be
   slow.

Quickstart
----------

Selecting a backend
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pychop
   pychop.backend("numpy")   # or "jax" or "torch"

Creating chopped arrays/tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from pychop import Chop
   from pychop.builtin import CPArray

   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
   A = CPArray(np.array([[1., 2.], [3., 4.]]), half)

Example: scalar outputs are CPFloat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pychop.builtin.linalg import det, norm, trace

   d = det(A)
   n = norm(A)
   t = trace(A)

   print(type(d), d)   # CPFloat
   print(float(n))     # convert to python float when needed

API reference
-------------

Eigen / decompositions
^^^^^^^^^^^^^^^^^^^^^

- ``eig(A, ...)``
- ``eigvals(A, ...)``
- ``eigh(A, ...)``
- ``eigvalsh(A, ...)``
- ``svd(A, ...)``
- ``qr(A, ...)``
- ``cholesky(A, ...)``

Solve / inverse
^^^^^^^^^^^^^^^

- ``solve(A, B, ...)``
- ``inv(A, ...)``
- ``pinv(A, ...)``

Scalar-returning routines (return ``CPFloat``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions return :class:`pychop.builtin.cpfloat.CPFloat` to preserve
chopped precision in scalar arithmetic chains:

- ``det(A, ...)``
- ``slogdet(A, ...)``
- ``matrix_rank(A, ...)``
- ``cond(A, ...)``
- ``norm(x, ...)``
- ``trace(A, ...)``

Array-returning utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``diagonal(A, ...)``

Matrix functions
^^^^^^^^^^^^^^^^

- ``expm(A, ...)``

  Backend mapping:

  - NumPy: SciPy-backed (``scipy.linalg.expm``)
  - JAX: ``jax.scipy.linalg.expm`` (when available)
  - Torch: ``torch.linalg.matrix_exp`` / ``torch.matrix_exp`` (version-dependent)

Advanced SciPy-backed functions (with host fallback)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``logm(A, *, allow_host_fallback=False, ...)``
- ``sqrtm(A, *, allow_host_fallback=False, ...)``
- ``polar(A, *, allow_host_fallback=False, ...)``

For NumPy backend these are SciPy-backed directly. For JAX/Torch backends these
raise an informative error by default, but can be enabled via host fallback.

Host fallback example (Torch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch, pychop
   from pychop import Chop
   from pychop.builtin import CPTensor
   from pychop.builtin.linalg import sqrtm, polar

   pychop.backend("torch")
   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

   A = CPTensor(torch.tensor([[2., 1.], [1., 2.]]), half)

   # Runs SciPy on CPU; returns CPTensor chopped+wrapped back
   S = sqrtm(A, allow_host_fallback=True)

   U, H = polar(A, allow_host_fallback=True)

Warnings
^^^^^^^^

- Host fallback transfers data to CPU and returns results in the current backend
  wrapper type. This can be slow and breaks accelerator placement.

LU factorization and decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch LU APIs differ from SciPy:

- SciPy and JAX typically provide LU as a triple ``(P, L, U)``.
- PyTorch commonly exposes factorization as ``(LU, pivots)``.

To handle this, two LU entry points are available:

``lu(A, ...)``
   Backend-specific LU:

   - NumPy: ``scipy.linalg.lu`` -> ``(P, L, U)``
   - JAX: ``jax.scipy.linalg.lu`` -> ``(P, L, U)``
   - Torch: ``torch.linalg.lu_factor`` -> ``(LU, pivots)``

``lu_factor(A, ...)``
   Returns backend-native factorization signature:

   - NumPy: ``scipy.linalg.lu_factor`` -> ``(lu, piv)``
   - Torch: ``torch.linalg.lu_factor`` -> ``(LU, pivots)``
   - JAX: not provided in this wrapper

``lu_plu(A, ...)``
   Returns a SciPy-like triple ``(P, L, U)`` on Torch, even if the local Torch
   version lacks ``torch.linalg.lu_unpack``. The implementation:

   1. Attempts ``torch.linalg.lu`` (if present and returns ``(P, L, U)``)
   2. Falls back to ``torch.lu`` (legacy) or ``torch.linalg.lu_factor``
   3. Manually unpacks the packed LU factors and pivot information

Example (Torch)
^^^^^^^^^^^^^^^

.. code-block:: python

   from pychop.builtin.linalg import lu_factor, lu_plu

   LU, piv = lu_factor(A)      # (LU, pivots)
   P, L, U = lu_plu(A)         # (P, L, U)

Implementation notes
--------------------

chopwrap_call and scalar handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The wrapper layer in ``pychop.builtin.dispatch`` drives the conversion:

- CP* inputs -> native arrays/tensors via the backend-specific unwrap
- Backend call
- Numeric array/tensor outputs -> wrapped back via backend-specific wrap
- Numeric scalar outputs -> controlled by ``scalar_mode``:

  - ``scalar_mode="cpfloat"`` returns :class:`pychop.builtin.cpfloat.CPFloat`
  - ``scalar_mode="python"`` returns a chopped Python scalar

Backend-aware CPFloat
^^^^^^^^^^^^^^^^^^^^^

``CPFloat`` uses the active backend to feed scalars into the correct chopper:

- NumPy backend: chops ``np.asarray(val)``
- Torch backend: chops a ``torch.as_tensor(val)``
- JAX backend: chops ``jax.numpy.asarray(val)``

This prevents mismatches such as passing NumPy arrays into a Torch chopper.

Troubleshooting
---------------

1. Host fallback returns NumPy arrays instead of CP wrappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a host fallback call returns ``numpy.ndarray``, ensure that the code path wraps
SciPy outputs using the backend spec with the known ``A.chopper`` (i.e., via
``spec.wrap(out_np, A.chopper)``) rather than calling ``chopwrap_call`` on a
pure NumPy output.

2. Torch LU unpack availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your Torch version does not provide ``torch.linalg.lu_unpack``, use
``lu_plu`` which includes a manual unpack fallback.

3. Performance considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- SciPy-on-host fallback is for research convenience, not performance.
- Prefer native backend implementations (JAX/Torch) when available.
