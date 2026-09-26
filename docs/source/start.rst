Installation and first computation
==================================

Install a released package with ``python -m pip install pychop``. To use changes
in a local checkout (including the new P3109 and time-series APIs), run from its
root:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

On Windows activate with ``.venv\Scripts\activate``. The package metadata declares
Python >=3.8. Dependency resolvers select compatible NumPy, pandas, SciPy,
scikit-learn and Dask versions. New examples use NumPy; optional frameworks are
not imported merely by importing Pychop.

Install only the extra needed for your application:

.. code-block:: bash

   python -m pip install -e '.[torch]'
   # Other choices: .[jax], .[tensorflow], .[all]

First quantization
------------------

.. code-block:: python

   import numpy as np
   import pychop
   from pychop import Chop

   pychop.backend("numpy")
   quantize = Chop(exp_bits=5, sig_bits=10, rmode=1)
   x = np.array([0.1, -0.3, 1.1], dtype=np.float64)
   y = quantize(x)
   print(y)
   print("absolute error:", np.abs(y - x))

``sig_bits`` counts trailing fraction bits; total precision includes one additional
implicit bit. ``rmode=1`` means nearest, ties to even. Use integer rounding modes
with ``Chop``; string modes belong to the separate :doc:`p3109` API.

Choose rounding boundaries explicitly
-------------------------------------

.. code-block:: python

   import numpy as np
   from pychop import Chop

   q = Chop(exp_bits=5, sig_bits=10)
   a = q(np.array([0.1, 0.2]))
   b = q(np.array([0.3, 0.4]))
   rounded_once = q(np.dot(a, b))
   products = q(a * b)
   rounded_each_step = q(q(products[0]) + products[1])
   print(rounded_once, rounded_each_step)

A matrix multiplication followed by a quantizer rounds the output; it does not
emulate every multiply and accumulator update. :doc:`builtin` provides wrappers
that apply quantization after supported operations. :doc:`linalg` discusses
algorithms with explicit precision transitions.

Backend selection
-----------------

The default is ``pychop.backend("auto")``: the input array type selects the
backend at call time. Explicit selection is process-global. Keep a quantizer
instance to preserve its random-number sequence. Recreating it with the same
``random_state`` restarts that sequence. Backend-specific random generators need
not produce identical sequences for the same seed.

P3109 automatically dispatches on NumPy, Torch, TensorFlow and JAX inputs,
independently of this global setting. It preserves the tensor backend and offers
explicit STE training gradients. The time-series calibration API remains NumPy
CPU; convert device data explicitly before using it.

Next steps
----------

* :doc:`p3109`: format selection, rounding, encode/decode and policy export.
* :doc:`timeseries`: fit, symbolize, reconstruct and reload calibration.
* :doc:`examples`: runnable quantized models and saved results.
* :doc:`validation`: correctness tests and reproducible CPU benchmarks.
