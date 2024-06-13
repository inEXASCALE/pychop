Start
=====================================================

The main function of the ``pychop`` is the method ``chop``, which is loaded by 

.. code:: python

    from pychop import chop


``pychop`` supports NumPy (default), JAX, Torch as backend for simulation. To set backend:

.. code:: python

    pychop.backend('numpy') # use NumPy as backend, other options: 'torch' and 'jax'
