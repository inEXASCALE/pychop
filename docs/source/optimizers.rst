.. _optimizers:

Quantized optimizers module
===========================

This module provides custom PyTorch optimizers with quantized momentum and accumulator states, designed for low-precision training simulations. These optimizers extend `torch.optim.Optimizer` and utilize the `Chop` or `Chopf` for quantization.

Classes
-------

.. class:: QuantizedSGD(params, lr=0.01, momentum=0.9, weight_decay=0, exp_bits=8, sig_bits=7, rmode=1)

   SGD optimizer with quantized momentum.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups.
   :type params: iterable
   :param lr: Learning rate (default: 0.01).
   :type lr: float
   :param momentum: Momentum factor (default: 0.9).
   :type momentum: float
   :param weight_decay: Weight decay (L2 penalty) (default: 0).
   :type weight_decay: float
   :param exp_bits: Number of exponent bits for quantization (default: 8).
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization (default: 7).
   :type sig_bits: int
   :param rmode: Rounding mode for quantization (e.g., 1, "nearest") (default: 1).
   :type rmode: int

   Quantizes the momentum buffer (if momentum > 0) and the parameter update.

   **Example:**

   .. code-block:: python

      optimizer = QuantizedSGD(model.parameters(), lr=0.01, momentum=0.9, rmode=1)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

.. class:: QuantizedRMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, exp_bits=8, sig_bits=7, rmode=1)

   RMSprop optimizer with quantized accumulator and optional momentum.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups.
   :type params: iterable
   :param lr: Learning rate (default: 0.01).
   :type lr: float
   :param alpha: Smoothing constant for accumulator (default: 0.99).
   :type alpha: float
   :param eps: Term added to denominator for numerical stability (default: 1e-8).
   :type eps: float
   :param weight_decay: Weight decay (L2 penalty) (default: 0).
   :type weight_decay: float
   :param momentum: Momentum factor (default: 0).
   :type momentum: float
   :param exp_bits: Number of exponent bits for quantization (default: 8).
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization (default: 7).
   :type sig_bits: int
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   Quantizes the square average accumulator and momentum buffer (if used), as well as the final update.

   **Example:**

   .. code-block:: python

      optimizer = QuantizedRMSprop(model.parameters(), lr=0.01, momentum=0.9, rmode=5)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

.. class:: QuantizedAdagrad(params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10, exp_bits=8, sig_bits=7, rmode=1)

   Adagrad optimizer with quantized accumulator.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups.
   :type params: iterable
   :param lr: Learning rate (default: 0.01).
   :type lr: float
   :param lr_decay: Learning rate decay (default: 0).
   :type lr_decay: float
   :param weight_decay: Weight decay (L2 penalty) (default: 0).
   :type weight_decay: float
   :param eps: Term added to denominator for numerical stability (default: 1e-10).
   :type eps: float
   :param exp_bits: Number of exponent bits for quantization (default: 8).
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization (default: 7).
   :type sig_bits: int
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   Quantizes the sum of squared gradients (accumulator) and the parameter update.

   **Example:**

   .. code-block:: python

      optimizer = QuantizedAdagrad(model.parameters(), lr=0.01, rmode=4)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

.. class:: QuantizedAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, exp_bits=8, sig_bits=7, rmode=1)

   Adam optimizer with quantized momentum and accumulator.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups.
   :type params: iterable
   :param lr: Learning rate (default: 1e-3).
   :type lr: float
   :param betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
   :type betas: tuple[float, float]
   :param eps: Term added to denominator for numerical stability (default: 1e-8).
   :type eps: float
   :param weight_decay: Weight decay (L2 penalty) (default: 0).
   :type weight_decay: float
   :param exp_bits: Number of exponent bits for quantization (default: 8).
   :type exp_bits: int
   :param sig_bits: Number of mantissa bits for quantization (default: 7).
   :type sig_bits: int
   :param rmode: Rounding mode for quantization (default: 1).
   :type rmode: int

   Quantizes the first moment (momentum), second moment (accumulator), and the parameter update.

   **Example:**

   .. code-block:: python

      optimizer = QuantizedAdam(model.parameters(), lr=0.001, rmode=6)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

Notes
-----

- All optimizers rely on the `Chop/Chopf` for quantization, which must be imported from its respective module.
- These optimizers are designed for low-precision training and may exhibit different convergence behavior compared to their full-precision counterparts.
