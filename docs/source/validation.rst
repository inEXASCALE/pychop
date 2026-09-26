Validation and performance
==========================

Correctness checks
------------------

From the repository root, using an editable installation:

.. code-block:: bash

   python -m pip install pytest gfloat
   python -m pytest tests/test_p3109.py tests/test_timeseries.py tests/test_chop_backend_reuse.py -q
   python -m pytest tests -q
   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs/source /tmp/pychop-docs

The gfloat comparison is optional in a core installation (those tests skip when
it is absent), and required in the dedicated numerical CI job. The P3109 suite
checks every code point for valid widths 3..8, round-trip encoding, official
Binary8p4 value tables, midpoint ties and adjacent float64 values, all five
shared deterministic modes and saturation, explicit-bit stochastic modes,
invalid configurations, noncontiguous/read-only inputs and wider formats.
Independent special-case tests cover unsigned negative values, round-to-odd,
canonical zero and propagation saturation. The implementation has no gfloat
runtime dependency. Differential tests also passed against upstream gfloat
revision ``caea0b1a971cc71fa9095336dd7457dc42b36dc7``.

Time-series tests verify known segment averages and bins, training/test separation,
constant inputs, invalid calibration, failed-fit state preservation and JSON
round trips. Application tests execute the toy models and reload their exports.
Backend reuse has a regression checking that stochastic sequences advance and
match explicit NumPy dispatch with the same seed.

CPU benchmark
-------------

.. code-block:: bash

   python benchmarks/benchmark_p3109.py --repeat 7 --output /tmp/p3109-benchmark.json

The benchmark checks exact results before timing. Both implementations receive
the same pre-generated float64 input and return float64 arrays. It compares
Pychop's public quantizer with ``gfloat.round_ndarray`` (not a scalar Python loop),
uses warm-ups, alternates timing order and reports median elapsed time. No JIT
compilation or random generation is included in either timed call. The JSON
records versions, machine, seed, distribution, sizes, samples and ratios.

The checked-in ``benchmarks/p3109_macos_arm64.json`` records a local macOS arm64
run against gfloat 0.5.2, NumPy 2.5.3 and Python 3.12. It covers Binary8p3se and
Binary8p4se, normal and wide-range inputs, and sizes 1, 1,024, 65,536 and 1,000,000.
Every measured case was faster than gfloat. Consult the JSON for exact timings;
ratios depend on the machine, load, input shape, dtype and policy. The benchmark
covers nearest-even without saturation; it is not a speed guarantee for other
modes, code conversion, GPU execution or complete applications.

Why the kernel is faster
------------------------

The dedicated NumPy path avoids generic array-namespace dispatch, repeated
format-property evaluation and many full-array intermediate selections. It uses
``frexp`` to determine binary exponents exactly, ``ldexp`` for power-of-two scaling,
and in-place updates to owned output buffers for saturation and signs. It does
not depend on unsafe fast-math, a lookup table proportional to ``2**k``, JIT
compilation or approximations to boundary decisions.

Limitations of local validation
--------------------------------

Core-only runs skip unavailable framework integrations. Passing NumPy tests does
not establish CUDA, TPU, TensorFlow, JAX or Torch behavior on every supported
version. Existing backend code remains separately tested. The package's declared
Python minimum and old dependency minima are not a claim that this local run
covered every historical environment; use the CI matrix and downstream tests.

Native tensor validation and timing
-----------------------------------

.. code-block:: bash

   python -m pip install -e '.[all]' pytest
   python -m pytest tests/test_p3109_backends.py -q
   python benchmarks/benchmark_p3109_backends.py --backend torch --compiled
   python benchmarks/benchmark_p3109_backends.py --backend jax --compiled
   python benchmarks/benchmark_p3109_backends.py --backend tensorflow --compiled

Native tests cover backend parity at code points and neighboring midpoints,
all deterministic and stochastic modes, all saturation policies, wider formats,
float32/64 and promoted half inputs, host subnormals, shapes, strided inputs,
compiled encoding/decoding, compiled random-bit rounding, STE gradients and
training/export examples. A CUDA check is skipped on hosts without CUDA.

The native benchmark uses float32 CPU arrays and validates against NumPy before
timing. It warms up compiled functions and synchronizes returned results on each
measurement. When a gfloat native backend exists (Torch/JAX), the same compilation
policy is applied to both candidates. TensorFlow has no gfloat native baseline,
so its benchmark reports latency without a gfloat speed ratio. Compilation time
is excluded, and results must not be extrapolated to untested accelerators.
``torch.compile`` in the benchmark uses the actual default optimizing compiler;
unit tests use the eager compiler backend to check complete graph capture
without requiring a C++ toolchain.

The native modules also include a dedicated PyTorch nearest-even path using
``frexp``/``ldexp`` to reduce eager tensor passes. Other tensor policies share the
IEEE bit-decomposition kernel. Both paths enforce the same host-format bounds.
On macOS, optimizing Torch compilation needs a working C++/OpenMP toolchain; a
missing standard header is a compiler configuration error, not an indication
that eager tensor quantization requires a compiler.

To exercise Torch's optimizing compiler in the graph/gradient regression test,
set ``PYCHOP_TORCH_COMPILE_BACKEND=inductor`` and run
``python -m pytest tests/test_p3109_backends.py -k 'torch and graph_and_ste'``.
The default setting is ``eager`` so ordinary tests do not require a compiler.

Recorded native CPU results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checked-in ``benchmarks/p3109_*_cpu.json`` files record this macOS arm64 run
for 65,536 float32 values, Binary8p4se and nearest-even, using nine alternating
samples after warm-up. Both Torch/JAX candidates use the same compilation policy.
These are observations on one CPU, not accelerator or all-format guarantees.

.. list-table:: Median latency (milliseconds)
   :header-rows: 1

   * - Execution
     - Pychop
     - gfloat
     - Speedup
   * - Torch 2.14.0 eager
     - 0.893
     - 1.747
     - 1.96x
   * - Torch 2.14.0 Inductor
     - 0.212
     - 0.265
     - 1.25x
   * - JAX 0.11.2 jit
     - 0.060
     - 0.233
     - 3.92x
   * - TensorFlow 2.21.0 tf.function
     - 0.734
     - Not measured
     - Not claimed

The local Torch Inductor run used Apple clang++, an explicit macOS SDK C++ header
path, and an ``OMP_PREFIX`` linking the same OpenMP library already loaded by
Torch. This avoided conflicting Homebrew/packaged OpenMP libraries. These were
per-process verification settings, not changes made by Pychop at import time.
