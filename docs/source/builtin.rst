.. _builtin-types:

=================================
Built-in low-precision types
=================================

``pychop`` ships three ready-made classes that automatically **chop to the
desired format after every arithmetic operation**:

* :class:`pychop.builtin.CPFloat` – scalar (Python ``float``-like)
* :class:`pychop.builtin.CPTensor` – :class:`torch.Tensor` subclass
* :class:`pychop.builtin.CPArray`  – :class:`numpy.ndarray` subclass

All three work with any ``LightChop`` (or ``Chop``) instance and keep the
result **inside the low-precision type**.

.. contents:: Contents
   :local:
   :backlinks: none


Quick import
============

.. code-block:: python

   from pychop import LightChop
   from pychop.builtin import CPFloat, CPTensor, CPArray

   pychop.backend('torch') # if using NumPy or JAX, switch to them correspondingly before the deployment

Common set-up
=============

.. code-block:: python

   # Half-precision (IEEE-754 binary16) – subnormals enabled
   half = LightChop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

   # Under-flow-free half (tiny numbers become zero)
   ufhalf = LightChop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)


Scalar – :class:`CPFloat`
=========================

.. autoclass:: pychop.builtin.CPFloat
   :members:
   :undoc-members:
   :show-inheritance:

**Example**

.. code-block:: python

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

.. autoclass:: pychop.builtin.CPTensor
   :members:
   :undoc-members:
   :show-inheritance:

**Example**

.. code-block:: python

   import torch

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


Under-flow-free (UF) formats
============================

Just create a ``LightChop`` with ``subnormal=False`` and pass it to any of the
three types:

.. code-block:: python

   uf = LightChop(exp_bits=5, sig_bits=10, subnormal=False, rmode=1)

   tiny = CPFloat(1e-40, uf)          # becomes 0.0 (flushed)
   print(tiny)                        # CPFloat(0.0, prec=uf)

   huge = CPTensor(torch.tensor([1e30, 1e35]), uf)
   print(huge)                        # CPTensor(tensor([1.0000e+30, inf]), ...)


Supported operations
====================

All Python arithmetic operators (``+ – * / // % **``) and the usual
library functions are dispatched through the subclass machinery:

* **NumPy** – any ufunc (``np.sin``, ``np.exp``, ``np.linalg.norm`` …)
* **PyTorch** – any ``torch.*`` function (``torch.nn.functional.relu``,
  ``torch.matmul``, ``torch.conv2d`` …)

The result is **always** chopped and returned as the same built-in type.

.. note::

   Reductions (``sum``, ``mean``, ``norm``) return a **scalar** of the same
   type when the input is a ``CPArray``/``CPTensor``.  If you need a plain
   Python number, call ``.item()`` or ``float(...)``.


Pickling / serialization
========================

All three classes implement ``__reduce_ex__`` and can be pickled/unpickled
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
* Use the **NumPy** backend (default) for pure-CPU workloads.

That’s it, simply drop the three imports into your code and you instantly get
**type-preserving low-precision arithmetic** for scalars, NumPy arrays and
PyTorch tensors!
