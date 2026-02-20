Mathematical functions 
========================================
 
``Pychop`` provides two ways to implement mathematical functions in reduced precision.  
The first approach requires specifying a backend, whereas the second approach does not require explicitly specifying one.


Mathematical functions I
------------------------------------------------------------------------

The `chop` class provides a suite of mathematical functions that operate on floating-point numbers with custom precision chopping. These functions apply the chopping mechanism (via `chop_wrapper`) to inputs and outputs, ensuring results adhere to the specified precision (e.g., fp16, fp32). Implementations are available for NumPy, PyTorch, and JAX, with slight variations noted below. Functions are categorized for clarity.
However, this method requires user to specify the backend first. 

.. note::
   - All functions use the `chop_wrapper` method to apply precision chopping before and after computation.
   - **NumPy**: Uses `numpy` (`np`) operations, operates on `np.ndarray`, and assumes a stateless implementation.
   - **PyTorch**: Uses `torch` operations, operates on `torch.Tensor`, and supports GPU acceleration.
   - **JAX**: Uses `jax.numpy` (`jnp`) operations, operates on `jax.Array`, requires a random key for chopping, and is JIT-compatible.
   - Examples assume a `chop` instance with half-precision (`prec='h'`) unless stated otherwise.


Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sin(x)

   Compute the sine of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped sine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

   **Example (NumPy):**

   .. code-block:: python

      import numpy as np
      chopper = chop(prec='h')
      x = np.array([0.0, 1.5708])  # ~[0, pi/2]
      result = chopper.sin(x)
      print(result)  # Expected: ~[0.0, 1.0] with chopping

   **Example (PyTorch):**

   .. code-block:: python

      import torch
      chopper = chop(prec='h')
      x = torch.tensor([0.0, 1.5708])
      result = chopper.sin(x)
      print(result)  # Expected: ~[0.0, 1.0] with chopping

   **Example (JAX):**

   .. code-block:: python

      import jax.numpy as jnp
      chopper = chop(prec='h')
      x = jnp.array([0.0, 1.5708])
      result = chopper.sin(x)
      print(result)  # Expected: ~[0.0, 1.0] with chopping

.. function:: cos(x)

   Compute the cosine of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped cosine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: tan(x)

   Compute the tangent of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped tangent of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: arcsin(x)

   Compute the arcsine of `x` with chopping. Input must be in [-1, 1].

   :param x: Input array/tensor in [-1, 1].
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped arcsine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is not in [-1, 1].

.. function:: arccos(x)

   Compute the arccosine of `x` with chopping. Input must be in [-1, 1].

   :param x: Input array/tensor in [-1, 1].
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped arccosine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is not in [-1, 1].

.. function:: arctan(x)

   Compute the arctangent of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped arctangent of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sinh(x)

   Compute the hyperbolic sine of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped hyperbolic sine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: cosh(x)

   Compute the hyperbolic cosine of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped hyperbolic cosine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: tanh(x)

   Compute the hyperbolic tangent of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped hyperbolic tangent of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: arcsinh(x)

   Compute the inverse hyperbolic sine of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped inverse hyperbolic sine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: arccosh(x)

   Compute the inverse hyperbolic cosine of `x` with chopping. Input must be >= 1.

   :param x: Input array/tensor (>= 1).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped inverse hyperbolic cosine of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is < 1.

.. function:: arctanh(x)

   Compute the inverse hyperbolic tangent of `x` with chopping. Input must be in (-1, 1).

   :param x: Input array/tensor in (-1, 1).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped inverse hyperbolic tangent of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is not in (-1, 1).

Exponential and logarithmic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: exp(x)

   Compute the exponential of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped exponential of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: expm1(x)

   Compute exp(x) - 1 with chopping, optimized for small `x`.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped exp(x) - 1.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: log(x)

   Compute the natural logarithm of `x` with chopping. Input must be positive.

   :param x: Input array/tensor (> 0).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped natural logarithm of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is <= 0.

.. function:: log10(x)

   Compute the base-10 logarithm of `x` with chopping. Input must be positive.

   :param x: Input array/tensor (> 0).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped base-10 logarithm of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is <= 0.

.. function:: log2(x)

   Compute the base-2 logarithm of `x` with chopping. Input must be positive.

   :param x: Input array/tensor (> 0).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped base-2 logarithm of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is <= 0.

.. function:: log1p(x)

   Compute log(1 + x) with chopping, optimized for small `x`. Input must be > -1.

   :param x: Input array/tensor (> -1).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped log(1 + x).
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is <= -1.

Power and root functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sqrt(x)

   Compute the square root of `x` with chopping. Input must be non-negative.

   :param x: Input array/tensor (>= 0).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped square root of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is < 0.

.. function:: cbrt(x)

   Compute the cube root of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped cube root of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

Miscellaneous functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: abs(x)

   Compute the absolute value of `x` with chopping.

   :param x: Input array/tensor (real or complex).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped absolute value of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: reciprocal(x)

   Compute the reciprocal (1/x) of `x` with chopping. Input must not be zero.

   :param x: Input array/tensor (!= 0).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped reciprocal of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :raises ValueError: If any element of `x` is 0.

.. function:: square(x)

   Compute the square of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped square of `x`.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

Additional mathematical functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: frexp(x)

   Decompose `x` into mantissa and exponent with chopping applied to mantissa.

   :param x: Input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Tuple of (chopped mantissa, exponent).
   :rtype: tuple (np.ndarray, np.ndarray) (NumPy), (torch.Tensor, torch.Tensor) (PyTorch), or (jax.Array, jax.Array) (JAX)

.. function:: hypot(x, y)

   Compute the Euclidean norm sqrt(x^2 + y^2) with chopping.

   :param x: First input array/tensor (real-valued).
   :param y: Second input array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :type y: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped Euclidean norm.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: diff(x, n=1)

   Compute the n-th order difference of `x` with chopping.

   :param x: Input array/tensor (real-valued).
   :param n: Order of difference (default: 1).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :type n: int
   :return: Chopped n-th order difference.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)

.. function:: power(x, y)

   Compute x raised to the power y with chopping.

   :param x: Base array/tensor (real-valued).
   :param y: Exponent array/tensor (real-valued).
   :type x: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :type y: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)
   :return: Chopped x^y.
   :rtype: np.ndarray (NumPy), torch.Tensor (PyTorch), or jax.Array (JAX)





Mathematical functions II
------------------------------------------------------------------------

The `pychop.math_func` module provides a suite of backend-aware mathematical functions that operate on floating-point numbers or arrays with custom precision chopping. These functions apply a provided `chop` callable to inputs and outputs, ensuring results adhere to the specified precision (e.g., fp16, fp32).  

Supported backends: NumPy, PyTorch, and JAX. Backend is inferred from the type of the input array/tensor. Functions are categorized for clarity.

.. note::
   - All functions apply the `chop` callable before and after computation.
   - Backend detection is automatic:
     - **NumPy**: `np.ndarray`
     - **PyTorch**: `torch.Tensor`
     - **JAX**: `jax.Array`
   - Inputs must satisfy domain constraints (e.g., positive for log, non-zero for reciprocal).
   - `matmul` requires inputs to be at least 1-dimensional; scalars are not allowed.


**Example (NumPy):**

.. code-block:: python

   import numpy as np
   import pychop.math_func as mf
   from pychop import Chop

   chopper = Chop(exp_bits=5, sig_bits=10, rmode=3)
   x = np.array([0.0, 1.5708])  # ~ [0, pi/2]
   result = mf.sin(x, chopper)
   print(result)  # Expected: ~ [0.0, 1.0] with chopping


**Example (PyTorch):**

.. code-block:: python

   import torch
   import pychop.math_func as mf
   from pychop import Chop

   chopper = Chop(exp_bits=5, sig_bits=10, rmode=3)
   x = torch.tensor([0.0, 1.5708])
   result = mf.sin(x, chopper)
   print(result)  # Expected: ~ [0.0, 1.0] with chopping


**Example (JAX):**

.. code-block:: python

   import jax.numpy as jnp
   import pychop.math_func as mf
   from pychop import Chop

   chopper = Chop(exp_bits=5, sig_bits=10, rmode=3)
   x = jnp.array([0.0, 1.5708])
   result = mf.sin(x, chopper)
   print(result)  # Expected: ~ [0.0, 1.0] with chopping


Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sin(x, chop)

   Compute sine of `x` with chopping.

   :param x: Real-valued input.
   :param chop: Callable that applies precision chopping.
   :return: Chopped sine of `x`.
   :rtype: Same type as `x` (NumPy, PyTorch, or JAX)

.. function:: cos(x, chop)
.. function:: tan(x, chop)
.. function:: arcsin(x, chop)
   Input must be in [-1, 1].
.. function:: arccos(x, chop)
   Input must be in [-1, 1].
.. function:: arctan(x, chop)

Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sinh(x, chop)
.. function:: cosh(x, chop)
.. function:: tanh(x, chop)
.. function:: arcsinh(x, chop)
.. function:: arccosh(x, chop)
   Input must be >= 1.
.. function:: arctanh(x, chop)
   Input must be in (-1, 1).

Exponential and logarithmic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: exp(x, chop)
.. function:: expm1(x, chop)
.. function:: log(x, chop)
   Input must be positive.
.. function:: log10(x, chop)
   Input must be positive.
.. function:: log2(x, chop)
   Input must be positive.
.. function:: log1p(x, chop)
   Input must be > -1.

Power and root functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sqrt(x, chop)
   Input must be non-negative.
.. function:: cbrt(x, chop)
.. function:: square(x, chop)
.. function:: power(x, y, chop)
   Compute x raised to the power y.

Arithmetic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: add(x, y, chop)
.. function:: subtract(x, y, chop)
.. function:: multiply(x, y, chop)
.. function:: divide(x, y, chop)
   Divisor must be non-zero.
.. function:: floor_divide(x, y, chop)
   Divisor must be non-zero.
.. function:: mod(x, y, chop)
   Divisor must be non-zero.

Linear algebra functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: dot(x, y, chop)
.. function:: matmul(x, y, chop)
   Inputs must be at least 1-dimensional; scalars are not allowed.

Reduction and aggregation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: sum(x, chop, axis=None)
.. function:: prod(x, chop, axis=None)
.. function:: mean(x, chop, axis=None)
.. function:: std(x, chop, axis=None)
.. function:: var(x, chop, axis=None)
.. function:: cumsum(x, chop, axis=None)
.. function:: cumprod(x, chop, axis=None)

Rounding functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: floor(x, chop)
.. function:: ceil(x, chop)
.. function:: round(x, chop)
.. function:: sign(x, chop)

Comparison functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: maximum(x, y, chop)
.. function:: minimum(x, y, chop)

Miscellaneous functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: frexp(x, chop)
   Returns tuple of (chopped mantissa, exponent).
.. function:: hypot(x, y, chop)
.. function:: diff(x, n=1, chop)
   Compute n-th order difference.
.. function:: reciprocal(x, chop)
   Input must be non-zero.

