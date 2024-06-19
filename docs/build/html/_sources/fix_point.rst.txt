Fixed point quantization
=====================================================

We start with a single or double precision (32 / 64 bit floating point) input X, 

The fixed point quantization demonstrates its superiority in U-Net image segmentation [1].
Following that, a basic bitwise shift quantization function is given by:

.. math::

    q(x) = \lfloor \texttt{clip}(x, 0, 2^b - 1) \ll b \rceil \gg b, 

where << and >> are left and right shift for bitwise operator, respectively. 

Then the given number $x$ to its fixed point value proceed by splitting its value into its fractional and integer parts:

.. math::

    x_f = \text{abs}(x) - \lfloor\text{abs}(x)\rfloor \quad \text{and} \quad x_i = \lfloor\text{abs}(x)\rfloor.


The fixed point representation for $x$ is given by 

.. math::

    Q_f{x} = \text{sign}(x) q(x_i) +  \text{sign}(x) q(x_f)



The usage is demonstrated step by step as below.

First we load the data in various format:

.. code:: python

    import numpy as np
    import torch
    import pychop
    from numpy import linalg
    import jax

    X_np = np.random.randn(500, 500) # Numpy array
    X_th = torch.Tensor(X_np) # torch array
    X_jx = jax.numpy.asarray(X_np)
    print(X_np)


The parameters that determine the fixed-point quantization is the following parameters

.. code-block:: language

    ibits : int, default=4
        The bitwidth of integer part. 

    fbits : int, default=4
        The bitwidth of fractional part. 


The backend of NumPy is performed by:

.. code:: python

    pychop.backend('numpy')
    pyq_f = pychop.fixed_point(ibits=4, fbits=4)
    pyq_f(X_np)

The backend of Torch is performed by:

.. code:: python

    pychop.backend('torch')
    pyq_f = pychop.fixed_point()
    pyq_f(X_th)

The backend of JAX is performed by:

.. code:: python

    pychop.backend('jax')
    pyq_f = pychop.fixed_point()
    pyq_f(X_jx)

[1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image
segmentation. In Medical Image Computing and Computer-Assisted Intervention, 234–241, 2015. Springer.


