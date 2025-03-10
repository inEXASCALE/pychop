Integer quantization
=====================================================

Integer quantization is another important feature of ``pychop``. It intention is to convert the floating point number into 
low bit-width integer, which speedup the computations in certain computing hardware. It performs quantization with 
user-defined bitwidths. The following example illustrates the usage of the method.

.. class:: Chopi(num_bits=8, symmetric=False, per_channel=False, channel_dim=0)

   A class for quantizing and dequantizing arrays to and from integer representations.

   This class supports both symmetric and asymmetric quantization, with optional per-channel quantization along a specified axis. It is designed for inference-style quantization in JAX, PyTorch, and NumPy frameworks, with framework-specific array types (``jnp.ndarray``, ``torch.Tensor``, ``np.ndarray``).

   :param int num_bits: Bit-width for quantization (e.g., 8 for INT8). Default is 8.
   :param bool symmetric: If True, use symmetric quantization (zero_point = 0). If False, use asymmetric quantization. Default is False.
   :param bool per_channel: If True, quantize per channel along the specified ``channel_dim``. If False, quantize the entire array. Default is False.
   :param int channel_dim: Dimension to treat as the channel axis for per-channel quantization. Default is 0.

   :ivar int qmin: Minimum quantized value (e.g., -128 for symmetric INT8, 0 for asymmetric INT8).
   :ivar int qmax: Maximum quantized value (e.g., 127 for INT8).
   :ivar scale: Scaling factor(s) for quantization, computed during calibration. Shape depends on ``per_channel`` (scalar or array).
   :ivar zero_point: Zero-point offset(s) for quantization, computed during calibration. None if symmetric, else matches ``scale`` shape.

   .. method:: calibrate(x)

      Calibrate the Chopi by computing the scale and zero-point based on the input array.

      :param x: Input array to calibrate from (``jnp.ndarray`` for JAX, ``torch.Tensor`` for PyTorch, ``np.ndarray`` for NumPy).
      :raises TypeError: If the input is not of the expected array type for the framework.

   .. method:: quantize(x)

      Quantize the input array to integers.

      If the Chopi has not been calibrated, it will automatically calibrate using the input array.

      :param x: Input array to quantize (``jnp.ndarray`` for JAX, ``torch.Tensor`` for PyTorch, ``np.ndarray`` for NumPy).
      :return: Quantized integer array (``jnp.ndarray`` with dtype ``int8`` for JAX, ``torch.Tensor`` with dtype ``torch.int8`` for PyTorch, ``np.ndarray`` with dtype ``int8`` for NumPy).
      :raises TypeError: If the input is not of the expected array type for the framework.

   .. method:: dequantize(q)

      Dequantize the integer array back to floating-point.

      :param q: Quantized integer array (``jnp.ndarray`` for JAX, ``torch.Tensor`` for PyTorch, ``np.ndarray`` for NumPy).
      :return: Dequantized floating-point array (``jnp.ndarray`` for JAX, ``torch.Tensor`` for PyTorch, ``np.ndarray`` for NumPy).
      :raises TypeError: If the input is not of the expected array type for the framework.
      :raises ValueError: If the Chopi has not been calibrated (i.e., ``scale`` is None).

   .. rubric:: Principle

   Quantization reduces the precision of floating-point values to integers to save memory and accelerate computation, especially on hardware with integer arithmetic support. The process involves:

   1. **Calibration**: Determine the range of the input array (min and max values) to compute a scaling factor (``scale``) and offset (``zero_point``).
   2. **Quantization**: Map floats to integers using ``q = round(x / scale + zero_point)``, clipped to ``[qmin, qmax]``.
   3. **Dequantization**: Recover approximate floats using ``x = (q - zero_point) * scale``.

   - **Symmetric**: Assumes ``zero_point = 0`` (e.g., range ``[-128, 127]`` for INT8), suitable for weights with zero-centered distributions.
   - **Asymmetric**: Allows ``zero_point`` to shift the range (e.g., ``[0, 255]`` for INT8), better for activations with non-zero minima.
   - **Per-channel**: Applies separate ``scale`` and ``zero_point`` per channel, improving accuracy for multi-channel data (e.g., CNN weights).

   .. rubric:: Examples

   **JAX Example**:

   .. code-block:: python

      import jax.numpy as jnp
      from pychop import Chopi  # Assuming module name
      pychop.backend('jax')

      x = jnp.array([[0.1, -0.2], [0.3, 0.4]])
      Chopi = Chopi(num_bits=8, symmetric=False)
      q = Chopi.quantize(x)
      dq = Chopi.dequantize(q)
      print(q)  # e.g., [[ 85  42] [106 127]], dtype=int8
      print(dq)  # e.g., [[ 0.098  -0.196] [ 0.294   0.392]]

   **PyTorch Example**:

   .. code-block:: python

      import torch
      from pychop import Chopi  # Assuming module name
      pychop.backend('torch')

      x = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
      Chopi = Chopi(num_bits=8, symmetric=False)
      q = Chopi.quantize(x)  # Inference mode
      dq = Chopi.dequantize(q)
      print(q)  # e.g., tensor([[ 85,  42], [106, 127]], dtype=torch.int8)
      print(dq)  # e.g., tensor([[ 0.098, -0.196], [ 0.294,  0.392]])

   **NumPy Example**:

   .. code-block:: python

      import numpy as np
      from pychop import Chopi  # Assuming module name
      pychop.backend('numpy')

      x = np.array([[0.1, -0.2], [0.3, 0.4]])
      Chopi = NumpyChopi(num_bits=8, symmetric=False)
      q = Chopi.quantize(x)
      dq = Chopi.dequantize(q)
      print(q)  # e.g., [[ 85  42] [106 127]], dtype=int8
      print(dq)  # e.g., [[ 0.098  -0.196] [ 0.294   0.392]]

   .. note::
      - The PyTorch version supports training mode via ``forward(x, training=True)`` for fake quantization, which isn’t shown here but is useful for quantization-aware training.
      - Exact integer values may vary slightly due to rounding and range differences.

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
    pyq_f = pychop.Chopi(bits=8) # The larger the ``bits`` are, the more accurate of the reconstruction is 
    X_q = pyq_f.quantize(X_np) # quant array -> integer
    X_inv = pyq_f.dequantize(X_q) # dequant array -> floating point values
    linalg.norm(X_inv - X_np)
        

    pychop.backend('torch')
    pyq_f = pychop.Chopi(bits=8)
    X_q = pyq_f.quantize(X_th)  # quant array -> integer
    X_inv = pyq_f.dequantize(X_q) # dequant array -> floating point values
    linalg.norm(X_inv - X_np)


    pychop.backend('jax')
    pyq_f = pychop.Chopi(bits=8)
    X_q = pyq_f.quantize(X_jx) # quant array -> integer
    X_inv = pyq_f.dequantize(X_q) # dequant array -> floating point values 
    linalg.norm(X_inv - X_jx)


