Examples
=====================================================

To import the package for NumPy backend:

.. code:: python

    from pychop.numpy import chop

or for Torch backend

.. code:: python

    from pychop.torch import chop


+-------------------------------------------------------+-------------------------------------------------------+
|                       format                          |                              description               |
+=======================================================+=======================================================+
|                    'q43', 'fp8-e4m3'     | NVIDIA quarter precision (4 exponent bits, 3 significand (mantissa) bits) |
+----------------------------------+----------------------------------+
| 'q52', 'fp8-e5m2'         | NVIDIA quarter precision (5 exponent bits, 2 significand bits) |
+----------------------------------+----------------------------------+
|  'b', 'bfloat16'          |       bfloat16 |
+----------------------------------+----------------------------------+
|  'h', 'half', 'fp16'      | IEEE half precision (the default) |
+----------------------------------+----------------------------------+
|  's', 'single', 'fp32'    | IEEE single precision |
+----------------------------------+----------------------------------+
|  'd', 'double', 'fp64'    | IEEE double precision |
+----------------------------------+----------------------------------+
|  'c', 'custom'            | custom format |
+----------------------------------+----------------------------------+


For Torch backend:

.. code:: python

    import torch
    from pychop.torch import chop

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.rand(size=(10000, 10000))
    tc = chop(prec='h', rmode=3, device=device) 
    y = tc.chop(x)
    print(y[0, :5])



For NumPy backend:

.. code:: python

    import numpy as np
    from pychop.numpy import chop

    x = np.random.rand(10000, 10000)
    nc = chop(prec='h', rmode=3) 
    y = nc.chop(x)
    print(y[0, :5])



One can also use customized floating point arithmetic:

First, define precisions:

.. code:: python

    from pychop import customs
    prec = customs(t=2, emax=10)


Second, define parameter ``customs`` instead of ``prec`` for both NumPy backend and Torch backend, 

.. code:: python

    x = np.random.rand(10000, 10000) # use x = torch.rand(size=(10000, 10000)) for Torch backend
    nc = chop(customs=prec, rmode=3, flip=0) 
    y = nc.chop(x)
    print(y[0, :5])


The above example is for bit-level simulation, you can depoy a direct setting to floating point arithmetic:

.. code:: python

    from pychop import simulate
    x = np.random.rand(10000, 10000) # use x = torch.rand(size=(10000, 10000)) for Torch backend
    si = simulate(base=2, t=11, emin=11, emax=22, sign=False, subnormal=False, rmode=1):
    y = si.rounding(x)
    print(y[0, :5])
