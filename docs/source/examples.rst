Basic examples
=====================================================

.. code:: python

    from pychop import chop


To set backend:

.. code:: python

    pychop.backend('numpy') # use NumPy as backend, other options: 'torch' and 'jax'

.. list-table:: Title
   :widths: 25 25
   :header-rows: 1

    * - format
      - description
    * - 'q43' and 'fp8-e4m3'
      - NVIDIA quarter precision (4 exponent bits, 3 significand (mantissa) bits)
    * - 'b' and 'bfloat16'
      -  bfloat16
    * - 'h' and 'half' and 'fp16' 
      - IEEE half precision (the default)
    * - 's' and 'single' and 'fp32'
      -  IEEE single precision
    * - 'd' and 'double' and 'fp64'
      - IEEE double precision
    * - 'c' and 'custom'
      - custom format



One can also use customized floating point arithmetic:

First, define precisions:

.. code:: python

    from pychop import customs
    prec = customs(t=2, emax=10)


Second, define parameter ``customs`` instead of ``prec`` for both NumPy backend and Torch backend, 

.. code:: python

    from pychop import chop
    x = np.random.rand(10000, 10000) # use x = torch.rand(size=(10000, 10000)) for Torch backend
    nc = chop(customs=prec, rmode=3, flip=0) 
    y = nc(x)
    print(y[0, :5])


The above example is for bit-level simulation, you can depoy a direct setting to floating point arithmetic:

.. code:: python

    from pychop import simulate
    import numpy as np
    x = np.random.rand(100, 100)
    si = simulate(base=2, t=11, emax=22, sign=False, subnormal=False, rmode=1)
    y = si.rounding(x)
    print(y[0, :5])

Note that if emin is not set, then IEEE 754 assumption is used which means emin = 1 - emax
