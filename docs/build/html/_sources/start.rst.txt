Start
=====================================================

The main function of the ``pychop`` is the method ``chop``, which is loaded by 

.. code:: python

    from pychop import chop



The other two dominant modules in ``pychop`` are  ``pychop.quant`` and ``pychop.fixed_point``, one can load them various


.. code:: python

    from pychop import quant
    from pychop import fpoint


    
``pychop`` supports NumPy (default), JAX, Torch as backend for simulation. Before performing the quantization, to set backend:

.. code:: python

    pychop.backend('numpy') # use NumPy as backend, other options: 'torch' and 'jax'
