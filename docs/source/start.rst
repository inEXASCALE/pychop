Quick start
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



.. note::

    Users do not need to specify the backend for precision emulation. By using ``pychop.backend("auto")`` (which is the default setting), ``pychop`` will automatically detect the required backend

    The difference between setting backend and without setting backend is that the speed of the first run is different, others are almost identical:

    .. code-block:: python

        import pychop
        from pychop import LightChop
        import numpy as np
        from time import time

        pychop.backend('numpy', 1) # Specify different backends, e.g., jax and torch
        np.random.seed(0)
        X = np.random.randn(5000, 5000) 

        ch = LightChop(exp_bits=5, sig_bits=10, rmode=3) # half precision

        st = time()
        X_q = ch(X)
        et = time()

        print("runtime: ", et - st)
        
    .. code-block:: bash

        Load NumPy backend.
        runtime:  0.65281081199646

    .. code-block:: python

        import pychop
        from pychop import LightChop
        import numpy as np
        from time import time

        X = np.random.randn(5000, 5000) 

        ch = LightChop(exp_bits=5, sig_bits=10, rmode=3) # half precision

        st = time()
        X_q = ch(X)
        et = time()

        print("runtime: ", et - st)
        print(X_q[:10, 0])


    .. code-block:: bash

        runtime:  0.9031667709350586