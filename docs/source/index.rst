Welcome to Pychop's documentation!
==================================

Using low precesion can gain extra speedup while resulting in less storage and energy cost. The intention of pychop, following the same function of chop in Matlab provided by Nick higham, is to simulate the low precision formats based on single and double precisions, which is pravalent on modern machine.

``pychop`` is a Python package for simulaing low precision in modern machine, it supports NumPy, Torch,  or JAX backend. 

The supported rounding modes include:

1. Round to nearest using round to even last bit to break ties
  (the default).

2. Round towards plus infinity (round up).

3. Round towards minus infinity (round down).

4. Round towards zero.

5. Stochastic rounding - round to the next larger or next smaller
  floating-point number with probability proportional to
  the distance to those floating-point numbers.

6. Stochastic rounding - round to the next larger or next smaller 
  floating-point number with equal probability.

Subnormal numbers is supported, they are flushed to zero if it not considered (by setting `subnormal` to 0).

Installation guide
------------------------------
``pychop`` has the following dependencies for its functionality:

   * numpy>=1.21
   * torch (only for torch use)
    
To install the current release via PIP use either of them according to one's need:

.. parsed-literal::
    
    pip install pychop (for Numpy backend)

    pip install torch-chop (for Torch backend)

.. admonition:: Note
   
   The documentation is still on going.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   symbols_machine_learning
   predict_with_symbols_representation
   api.rst
   license



