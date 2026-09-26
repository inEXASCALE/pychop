Runnable application examples
=============================

NumPy quantization and linear inference
---------------------------------------

From an editable installation at the repository root:

.. code-block:: bash

   python examples/p3109/quickstart.py

This self-contained program uses synthetic NumPy data and needs no downloaded
dataset, plotting library, GPU or training framework. It demonstrates rounding,
integer encoding and decoding, JSON-compatible policy export, reproducible
stochastic rounding with explicit random bits, and a two-input linear model.
The model quantizes its weights and inputs, accumulates the matrix product in
host precision, then quantizes its output. Choose those rounding boundaries
explicitly when comparing numerical error.

.. literalinclude:: ../../examples/p3109/quickstart.py
   :language: python
   :linenos:

For writing and reloading a policy together with integer model weights, see
:doc:`p3109`. The native training example below includes an executable export
and reload workflow for each supported tensor framework.

Other numerical workflows
--------------------------

The existing ``examples/mixed_precision/`` directory contains iterative
refinement, conjugate gradients, GMRES, LU/QR precision switching and residual
correction. See its README for individual commands. The ``examples/algorithms/``
directory contains additional small solvers. These demonstrate host computations
with explicit quantization, rather than native low-bit BLAS kernels.

Neural-network examples need the associated optional backend, and some older
examples require datasets or training dependencies. ``examples/deprecated/`` is
historical material, not the supported quick-start path.

Native tensor training and export
---------------------------------

Install the matching optional backend, then run one of:

.. code-block:: bash

   python examples/p3109/tensor_training.py --backend torch --output-dir /tmp/p3109-torch
   python examples/p3109/tensor_training.py --backend tensorflow --output-dir /tmp/p3109-tf
   python examples/p3109/tensor_training.py --backend jax --output-dir /tmp/p3109-jax

Each trains the same three-input linear model on synthetic data for 25 steps,
using explicit P3109 STE quantization of weights and predictions. TensorFlow and
JAX training steps are compiled. The example saves ``policy.json`` and
``model.npz`` with integer weight codes and loss history. It verifies native
code round trips and evaluates the exported model using the NumPy decoder,
illustrating portability between training frameworks and a CPU deployment.

.. literalinclude:: ../../examples/p3109/tensor_training.py
   :language: python
