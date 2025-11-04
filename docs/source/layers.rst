.. _layers:

Quantized layers module
=======================

This module provides custom PyTorch neural network layers with quantized computations, designed for low-precision simulations. These layers extend `torch.nn.Module` and utilize the `FloatPrecisionSimulator` for quantization.

Classes
-------

.. class:: QuantizedLinear(in_features, out_features, exp_bits, sig_bits, rmode=1)

   Fully connected layer with quantized weights, bias, and inputs.

   :param in_features: Size of each input sample.
   :type in_features: int
   :param out_features: Size of each output sample.
   :type out_features: int
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      layer = QuantizedLinear(10, 5, exp_bits=5, sig_bits=10, rmode=1)
      x = torch.randn(2, 10)
      y = layer(x)  # Shape: [2, 5]

.. class:: QuantizedMultiheadAttention(embed_dim, num_heads, exp_bits, sig_bits, dropout=0.0, rmode=1)

   Multi-head attention layer with quantized projections and outputs.

   :param embed_dim: Total dimension of the model.
   :type embed_dim: int
   :param num_heads: Number of parallel attention heads.
   :type num_heads: int
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param dropout: Dropout probability (default: 0.0).
   :type dropout: float
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      attn = QuantizedMultiheadAttention(512, 8, exp_bits=5, sig_bits=10)
      x = torch.randn(2, 10, 512)
      y, weights = attn(x, x, x)  # y: [2, 10, 512], weights: attention weights

.. class:: QuantizedLinearNorm(normalized_shape, exp_bits, sig_bits, eps=1e-5, rmode=1)

   Layer normalization with quantized inputs, weights, and biases.

   :param normalized_shape: Shape to normalize over (int or tuple).
   :type normalized_shape: int or tuple
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param eps: Small value added for numerical stability (default: 1e-5).
   :type eps: float
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      norm = QuantizedLinearNorm(512, exp_bits=5, sig_bits=10, rmode="towards_zero")
      x = torch.randn(2, 10, 512)
      y = norm(x)  # Shape: [2, 10, 512]

.. class:: QuantizedConv2d(in_channels, out_channels, kernel_size, exp_bits, sig_bits, stride=1, padding=0, rmode=1)

   2D convolution layer with quantized weights, bias, and inputs.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param out_channels: Number of output channels.
   :type out_channels: int
   :param kernel_size: Size of the convolution kernel (int or tuple).
   :type kernel_size: int or tuple
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param stride: Stride of the convolution (default: 1).
   :type stride: int or tuple
   :param padding: Padding added to inputs (default: 0).
   :type padding: int or tuple
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      conv = QuantizedConv2d(3, 16, 3, exp_bits=5, sig_bits=10)
      x = torch.randn(2, 3, 32, 32)
      y = conv(x)  # Shape: [2, 16, 30, 30]

.. class:: QuantizedRNN(input_size, hidden_size, exp_bits, sig_bits, num_layers=1, bias=True, nonlinearity="tanh", rmode=1)

   RNN layer with quantized weights and hidden states.

   :param input_size: Size of input features.
   :type input_size: int
   :param hidden_size: Size of hidden state.
   :type hidden_size: int
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param num_layers: Number of recurrent layers (default: 1).
   :type num_layers: int
   :param bias: If True, adds bias terms (default: True).
   :type bias: bool
   :param nonlinearity: Activation function ("tanh" or "relu") (default: "tanh").
   :type nonlinearity: str
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      rnn = QuantizedRNN(10, 20, exp_bits=5, sig_bits=10)
      x = torch.randn(2, 5, 10)
      y, h = rnn(x)  # y: [2, 5, 20], h: final hidden state

.. class:: QuantizedLSTM(input_size, hidden_size, exp_bits, sig_bits, num_layers=1, bias=True, rmode=1)

   LSTM layer with quantized weights and states.

   :param input_size: Size of input features.
   :type input_size: int
   :param hidden_size: Size of hidden state.
   :type hidden_size: int
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param num_layers: Number of recurrent layers (default: 1).
   :type num_layers: int
   :param bias: If True, adds bias terms (default: True).
   :type bias: bool
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      lstm = QuantizedLSTM(10, 20, exp_bits=5, sig_bits=10)
      x = torch.randn(2, 5, 10)
      y, (h, c) = lstm(x)  # y: [2, 5, 20], h/c: final hidden/cell states

.. class:: QuantizedGRU(input_size, hidden_size, exp_bits, sig_bits, num_layers=1, bias=True, rmode=1)

   GRU layer with quantized weights and hidden states.

   :param input_size: Size of input features.
   :type input_size: int
   :param hidden_size: Size of hidden state.
   :type hidden_size: int
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param num_layers: Number of recurrent layers (default: 1).
   :type num_layers: int
   :param bias: If True, adds bias terms (default: True).
   :type bias: bool
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      gru = QuantizedGRU(10, 20, exp_bits=5, sig_bits=10)
      x = torch.randn(2, 5, 10)
      y, h = gru(x)  # y: [2, 5, 20], h: final hidden state

.. class:: QuantizedBatchNorm2d(num_features, exp_bits, sig_bits, eps=1e-5, momentum=0.1, rmode=1)

   2D batch normalization with quantized inputs and parameters.

   :param num_features: Number of features (channels).
   :type num_features: int
   :param exp_bits: Number of exponent bits for quantization.
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization.
   :type sig_bits: int
   :param eps: Small value added for numerical stability (default: 1e-5).
   :type eps: float
   :param momentum: Momentum for running mean/variance (default: 0.1).
   :type momentum: float
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   **Example:**

   .. code-block:: python

      bn = QuantizedBatchNorm2d(3, exp_bits=5, sig_bits=10)
      x = torch.randn(2, 3, 32, 32)
      y = bn(x)  # Shape: [2, 3, 32, 32]

Notes
-----

- These layers are designed for low-precision inference and training, potentially affecting numerical precision and convergence.
