Runnable application examples
=============================

From an editable installation at the repository root:

.. code-block:: bash

   python examples/p3109/quickstart.py
   python examples/p3109/toy_models.py --output-dir /tmp/pychop-demo

Both programs are self-contained, deterministic and use synthetic NumPy data.
They need no downloaded dataset, plotting library, GPU or training framework.

The combined example builds three signals (noisy sine, AR(1), damped oscillator)
and three toy computations:

1. An AR(1) coefficient fitted by least squares, quantized to Binary8p4, and used
   for one-step prediction with quantized inputs and outputs.
2. An explicit-Euler damped oscillator with quantized state updates, compared
   against the same integration steps in host precision. Matrix products use
   host accumulation before output quantization.
3. A nearest-centroid classifier over symbolic windows, using training-only
   centroids and held-out sine/oscillator windows. Symbol indices are ordinal
   features here; this deliberately simple baseline is not a claim of a
   statistically reliable classifier.

Output files
------------

``parameters.json`` contains both the P3109 policy and fitted symbolizer, plus
full-precision and quantized AR coefficients. ``results.npz`` contains P3109
integer code points, time-series symbols, reconstructions, predictions and the
oscillator trajectory. ``metrics.json`` contains per-signal errors and classifier
accuracy. Files in the selected directory are replaced on subsequent runs.

The program reloads JSON and NPZ (with ``allow_pickle=False``) and verifies exact
reproduction of the encoded data and symbols. In the recorded local run the sine,
AR(1) and oscillator quantization RMSEs were approximately 0.0148, 0.00562 and
0.00742. The classifier accuracy was 0.5; this baseline illustrates the API and
also shows that a compact representation alone does not ensure prediction quality.

Full combined program
---------------------

.. literalinclude:: ../../examples/p3109/toy_models.py
   :language: python
   :linenos:

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
