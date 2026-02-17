.. Copyright (c) 2024 Graphcore Ltd. All rights reserved.

.. note::

  Check the version number of this documentation against the `gfloat` version
  you are using.  "Latest" refers to the head on https://github.com/graphcore-research/gfloat,
  while pypi versions installed using `pip install` will have corresponding `vX.Y.Z` tags.

GFloat: Generic floating point formats in Python
================================================

GFloat is designed to allow experimentation with a variety of floating-point
formats in Python.  Headline features:

  * A wide variety of floating point formats defined in :py:class:`gfloat.formats`

    - IEEE 754 and P3109, BFloat, OCP FP8 and MX

  * Conversion between floats under numerous rounding modes

    - Scalar code is optimized for readability
    - Array code is faster, and can operate on Numpy, JAX, or PyTorch arrays.

  * Notebooks useful for teaching and exploring float formats

Provided Formats
----------------

Formats are parameterized by the primary parameters of:

  * Width in bits (k)
  * Precision (p)
  * Exponent bias (bias)

with additional fields defining the presence/encoding of:

  * Domain (Finite vs Extended)
  * Signed/unsigned
  * Not-a-number (NaN) values
  * Negative zero
  * Subnormal numbers
  * Two's complement encoding (of the significand)

This allows an implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - `IEEE 754 <https://en.wikipedia.org/wiki/IEEE_754>`_: Binary16, Binary32
 - `Brain floating point <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_: BFloat16
 - |p3109_link|: P3109_{K}p{P}{s}{d} for K > 2, and 1 <= P <= K, Signedness, and Domain
 - |ocp_link|: E5M2, E4M3
 - Types from the |ocp_mx_link| spec: E8M0, INT8, and FP4, FP6 types

As well as block formats from |ocp_mx_link|.

.. |ocp_mx_link| raw:: html

   <a href="https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf" target="_blank">
     OCP MX
   </a>

.. |ocp_link| raw:: html

   <a href="https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf" target="_blank">
     OCP Float8
   </a>

.. |p3109_link| raw:: html

    <a href="https://github.com/P3109/Public" target="_blank">
      IEEE P3109
    </a>

Rounding modes
--------------

Various rounding modes:
    * Directed modes: Toward Zero, Toward Positive, Toward Negative
    * Round-to-nearest, with Ties to Even or Ties to Away
    * Stochastic rounding, with specified numbers of random bits


See Also
--------

GFloat, being a pure Python library, favours readability and extensibility over speed
(although the `*_ndarray` functions are reasonably fast for large arrays).
For fast implementations of these datatypes see, for example,
`ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_,
`bitstring <https://github.com/scott-griffiths/bitstring>`_,
`MX PyTorch Emulation Library <https://github.com/microsoft/microxcaling>`_,
`APyTypes <https://apytypes.github.io/apytypes>`_,
`TorchAO <https://github.com/pytorch/ao>`_.

To get started with the library, we recommend perusing the notebooks,
otherwise you may wish to jump straight into the API.

Contents
========

.. toctree::
   :hidden:

   self

.. toctree::

   notebooks
   api
   formats


Index and Search
================

* :ref:`genindex`
* :ref:`search`
