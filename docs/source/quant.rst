Integer quantization
=====================================================

Integer quantization is another important feature of ``pychop``. It intention is to convert the floating point number into 
low bit-width integer, which speedup the computations in certain computing hardware. It performs quantization with 
user-defined bitwidths. The following example illustrates the usage of the method.


.. code:: python

    import numpy as np
    import torch
    import pychop
    from numpy import linalg
    import jax

    X_np = np.random.randn(500, 500) # NumPy array
    X_th = torch.Tensor(X_np) # Torch array
    X_jx = jax.numpy.asarray(X_np) # JAX array
    print(X_np)

    pychop.backend('numpy')
    pyq_f = pychop.quant(bits=8) # The larger the ``bits`` are, the more accurate of the reconstruction is 
    X_q = pyq_f(X_np) # quant array -> integer
    X_inv = pyq_f.dequant(X_q) # dequant array -> floating point values
    linalg.norm(X_inv - X_np)
        

    pychop.backend('torch')
    pyq_f = pychop.quant(bits=8)
    X_q = pyq_f(X_th)  # quant array -> integer
    X_inv = pyq_f.dequant(X_q) # dequant array -> floating point values
    linalg.norm(X_inv - X_np)


    pychop.backend('jax')
    pyq_f = pychop.quant(bits=8)
    X_q = pyq_f(X_jx) # quant array -> integer
    X_inv = pyq_f.dequant(X_q) # dequant array -> floating point values 
    linalg.norm(X_inv - X_jx)


One can also load the required parameters via:

.. code:: python

    print(pyq_f.scaling)
    print(pyq_f.zpoint)


Also to perform a symmetric quantization, you can use:


.. code:: python

    pyq_f = pychop.quant(bits=8, zpoint=0) # by setting zpoint=0


By using unsign quantization, set parameter ``sign=0``, use

.. code:: python
    
    pyq_f = pychop.quant(bits=8, sign=0) # by setting zpoint=0
