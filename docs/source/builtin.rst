.. _builtin-types:

=================================
Built-in low-precision types
=================================

``Pychop`` ships four ready-made classes that automatically **chop to the
desired format after each public operation boundary**:

* :class:`pychop.builtin.CPFloat` – scalar (Python ``float``-like)
* :class:`pychop.builtin.CPArray`  – :class:`numpy.ndarray` subclass
* :class:`pychop.builtin.CPJaxArray` – JAX array wrapper
* :class:`pychop.builtin.CPTensor` – :class:`torch.Tensor` subclass

All four work with any ``Chop`` (or ``FaultChop``) instance and keep the
result **inside the low-precision type** when the wrapped operation returns a
floating-point or complex value. Integer and Boolean outputs, such as pivot
indices and comparison masks, are left as ordinary backend values.

.. note::

   The built-in types chop after Python/NumPy/PyTorch/JAX operation calls. They
   do not chop every internal multiply/add inside fused BLAS, LAPACK, XLA, or
   PyTorch kernels. For strict per-arithmetic-step simulation, write the
   algorithm steps explicitly with the built-in types or use a specialized
   chopped kernel.


Quick import
============

.. code-block:: python

   import pychop
   from pychop import Chop
   from pychop.builtin import CPFloat, CPArray, cast_precision
   # Import CPTensor or CPJaxArray only after installing the corresponding backend.

   pychop.backend('torch') # use 'numpy', 'jax', or 'torch' for the matching container

Common set-up
=============

.. code-block:: python

   # Half-precision (IEEE-754 binary16) – subnormals enabled
   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

   # Under-flow-free half (tiny numbers become zero)
   ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)


Scalar – :class:`CPFloat`
=========================

.. autoclass:: pychop.builtin.CPFloat
   :members:
   :undoc-members:
   :show-inheritance:

**Example**

.. code-block:: python

   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
   ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)
   
   a = CPFloat(1.234567, half)
   b = CPFloat(0.987654, half)

   print(a)                     # CPFloat(1.23438, prec=half)
   c = a + b                    # stays a CPFloat, chopped
   print(c)                     # CPFloat(2.22203, prec=half)
   d = a * b / 2.0
   print(d)                     # CPFloat(0.609863, prec=half)

   # mixed with a normal Python float
   e = a + 3.14
   print(e)                     # CPFloat(4.37438, prec=half)


PyTorch – :class:`CPTensor`
============================

.. py:class:: pychop.builtin.CPTensor(data, chopper)

   Optional framework container. Install its backend before importing this class.

**Example**

.. code-block:: python

   import torch
   from pychop.builtin import CPTensor

   pychop.backend('torch')  # Switch to Torch backend
   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
   ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)

   x = CPTensor(torch.tensor([1.1, 2.2, 3.3]), half)
   y = CPTensor(torch.tensor([0.5, 1.5, 2.5]), half)

   print(x)                     # CPTensor(tensor([1.1, 2.2, 3.3]), device=cpu, prec=half)
   z = x + y
   print(z)                     # CPTensor(tensor([1.6, 3.7, 5.8]), device=cpu, prec=half)

   # broadcasting with a plain tensor
   reg = torch.tensor([10.0, 20.0, 30.0])
   w = x * reg
   print(w)                     # CPTensor(tensor([11.0, 44.0, 99.0]), device=cpu, prec=half)

   # matrix multiplication
   A = CPTensor(torch.randn(4, 3), half)
   B = CPTensor(torch.randn(3, 5), half)
   C = A @ B
   print(C.shape)               # torch.Size([4, 5])

   # GPU works out-of-the-box
   if torch.cuda.is_available():
      A = A.to('cuda')
      B = B.to('cuda')
      C = A @ B
      print(C.device)          # cuda:0


NumPy – :class:`CPArray`
========================

.. autoclass:: pychop.builtin.CPArray
   :members:
   :undoc-members:
   :show-inheritance:

**Example**

.. code-block:: python

   import numpy as np

   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
   ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)
   p = CPArray(np.array([10.0, 20.0, 30.0]), half)
   q = CPArray(np.array([1.0, 2.0, 3.0]), half)

   print(p)                     # CPArray([10. 20. 30.], prec=half)
   r = p - q
   print(r)                     # CPArray([ 9. 18. 27.], prec=half)

   # element-wise with a normal ndarray
   plain = np.array([0.5, 1.5, 2.5])
   s = p * plain
   print(s)                     # CPArray([ 5. 30. 75.], prec=half)

   # linear-algebra (still chopped)
   M = CPArray(np.random.rand(3, 4), half)
   N = CPArray(np.random.rand(4, 2), half)
   P = M @ N
   print(P.shape)               # (3, 2)



JAX – :class:`CPJaxArray`
=========================

.. py:class:: pychop.builtin.CPJaxArray(data, chopper)

   Optional framework container. Install its backend before importing this class.

**Example**

.. code-block:: python

   from pychop.builtin import CPJaxArray

   pychop.backend('jax')  # Switch to JAX backend
   half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)
   ufhalf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)

   x = CPJaxArray(jnp.array([1.1, 2.2, 3.3]), half)
   y = CPJaxArray(jnp.array([0.5, 1.5, 2.5]), half)
   print(x)                     # CPJaxArray([1.1, 2.2, 3.3], prec=half)
   z = x + y
   print(z)                     # CPJaxArray([1.6, 3.7, 5.8], prec=half)
   # broadcasting with a plain array
   reg = jnp.array([10.0, 20.0, 30.0])
   w = x * reg
   print(w)                     # CPJaxArray([11.0, 44.0, 99.0], prec=half)
   # matrix multiplication

   gpu_devices = jax.devices('gpu')
   if gpu_devices:
      with jax.default_device(gpu_devices[0]):
                     
         A = CPJaxArray(jax.random.normal(jax.random.PRNGKey(0), (4, 3)), half)
         B = CPJaxArray(jax.random.normal(jax.random.PRNGKey(1), (3, 5)), half)
         C = A @ B
         print(C.shape)               # (4, 5)
         # GPU works out-of-the-box (JAX auto-dispatches)
         print(C.to_regular().devices())  # {CpuDevice()} or {CudaDevice()}


Under-flow-free (UF) formats
============================

Just create a ``Chop`` with ``subnormal=False`` and pass it to any of the
four types:

.. code-block:: python

   uf = Chop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)

   tiny = CPFloat(1e-40, uf)          # becomes 0.0 (flushed)
   print(tiny)                        # CPFloat(0.0, prec=uf)

   huge = CPTensor(torch.tensor([1e30, 1e35]), uf)
   print(huge)                        # CPTensor(tensor([1.0000e+30, inf]), ...)


Supported operations
====================

All Python arithmetic operators (``+ – * / // % **``) and many library
functions are dispatched through the wrapper/subclass machinery:

* **NumPy** – any ufunc (``np.sin``, ``np.exp``, ``np.linalg.norm`` …)
* **JAX** – operators implemented by :class:`CPJaxArray`; use
  :mod:`pychop.builtin.linalg` or ``chopwrap`` for wrapped high-level JAX
  outputs
* **PyTorch** – any ``torch.*`` function (``torch.nn.functional.relu``,
  ``torch.matmul``, ``torch.conv2d`` …)

Floating-point and complex results are chopped and returned as the matching
built-in type. Non-floating outputs are returned as ordinary backend values.

.. note::

   Scalar-returning wrappers in :mod:`pychop.builtin.linalg` return
   :class:`CPFloat` where scalar chaining is useful. Backend-native reductions
   may return backend scalars or arrays depending on the library.


Switching precision
===================

Use :func:`pychop.builtin.cast_precision` to switch a value to a new ``Chop``
configuration. Casting chops immediately and preserves the container family.

.. autofunction:: pychop.builtin.cast_precision

.. code-block:: python

   fp8 = Chop(exp_bits=4, sig_bits=3, subnormal=True, rmode=1)
   x_fp8 = cast_precision(x, fp8)


Pickling / serialization
========================

The NumPy and PyTorch built-in containers implement ``__reduce_ex__`` and can be pickled/unpickled
with the usual ``pickle`` module.

.. code-block:: python

   import pickle, io

   buf = io.BytesIO()
   pickle.dump(a, buf)          # a is a CPFloat
   buf.seek(0)
   a2 = pickle.load(buf)
   print(a2)                    # same value & chopper


Performance tip
===============

* Use the **PyTorch** backend (``pychop.backend('torch')``) for GPU-accelerated
  chopping.
* Use the **TensorFlow** backend (``pychop.backend('tensorflow')``) for TensorFlow/Keras
  workflows with STE-based gradient support. TensorFlow does not currently have
  a built-in ``CP*`` wrapper type.
* Use the **NumPy** backend for NumPy inputs (auto dispatch is the default) for pure-CPU workloads.

That’s it, simply drop the imports into your code and you get
**type-preserving low-precision arithmetic** for scalars, NumPy arrays, JAX
arrays, and PyTorch tensors.
