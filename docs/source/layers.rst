.. _pychop.layers:

Quantized layers module
=======================

.. py:module:: pychop.layers

This module provides **drop-in quantized layer replacements** for PyTorch models
to support both floating-point and integer **quantization-aware training (QAT)**.

All classes follow the same API as their original :mod:`torch.nn` counterparts.
When a ``chop`` quantizer (with STE) is provided, weights and activations are
fake-quantized during the forward pass while gradients flow through unchanged
(Straight-Through Estimator).

Three STE-enabled quantizers are provided:

* ``ChopSTE`` — floating-point (exponent + significand)
* ``ChopfSTE`` — fixed-point (integer + fractional bits)
* ``ChopiSTE`` — integer (uniform or symmetric)

STE Quantizers (Core)
---------------------

.. autoclass:: ChopSTE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ChopfSTE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ChopiSTE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FakeQuantizeSTE
   :members:

.. autoclass:: FakeFQuantizeSTE
   :members:

.. autoclass:: FakeIQuantizeSTE
   :members:

Utility Functions
-----------------

.. autofunction:: post_quantization

Floating-Point / Fixed-point Quantized Layers (QAT)
=====================================

These layers use ``ChopSTE`` (or ``Chop``) for **floating-point QAT**.

.. autoclass:: QuantizedLinear
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantizedConv1d
   :members:

.. autoclass:: QuantizedConv2d
   :members:

.. autoclass:: QuantizedConv3d
   :members:

.. autoclass:: QuantizedConvTranspose1d
   :members:

.. autoclass:: QuantizedConvTranspose2d
   :members:

.. autoclass:: QuantizedConvTranspose3d
   :members:

.. autoclass:: QuantizedRNN
   :members:

.. autoclass:: QuantizedLSTM
   :members:

.. autoclass:: QuantizedGRU
   :members:

.. autoclass:: QuantizedMaxPool1d
   :members:

.. autoclass:: QuantizedMaxPool2d
   :members:

.. autoclass:: QuantizedMaxPool3d
   :members:

.. autoclass:: QuantizedAvgPool1d
   :members:

.. autoclass:: QuantizedAvgPool2d
   :members:

.. autoclass:: QuantizedAvgPool3d
   :members:

.. autoclass:: QuantizedAdaptiveAvgPool2d
   :members:

.. autoclass:: QuantizedBatchNorm1d
   :members:

.. autoclass:: QuantizedBatchNorm2d
   :members:

.. autoclass:: QuantizedBatchNorm3d
   :members:

.. autoclass:: QuantizedLayerNorm
   :members:

.. autoclass:: QuantizedInstanceNorm1d
   :members:

.. autoclass:: QuantizedInstanceNorm2d
   :members:

.. autoclass:: QuantizedInstanceNorm3d
   :members:

.. autoclass:: QuantizedGroupNorm
   :members:

.. autoclass:: QuantizedMultiheadAttention
   :members:

.. autoclass:: QuantizedEmbedding
   :members:

**Convenience aliases**

.. autodata:: QuantizedAttention

.. autodata:: QuantizedAvgPool

Activation & Regularization Layers (Floating-Point)
---------------------------------------------------

.. autoclass:: QuantizedReLU
   :members:

.. autoclass:: QuantizedLeakyReLU
   :members:

.. autoclass:: QuantizedSigmoid
   :members:

.. autoclass:: QuantizedTanh
   :members:

.. autoclass:: QuantizedGELU
   :members:

.. autoclass:: QuantizedELU
   :members:

.. autoclass:: QuantizedSiLU
   :members:

.. autoclass:: QuantizedPReLU
   :members:

.. autoclass:: QuantizedSoftmax
   :members:

.. autoclass:: QuantizedDropout
   :members:

Integer Quantized Layers (QAT)
==============================

These layers use ``ChopiSTE`` for **integer QAT** (uniform or symmetric).

.. autoclass:: IQuantizedLinear
   :members:

.. autoclass:: IQuantizedConv1d
   :members:

.. autoclass:: IQuantizedConv2d
   :members:

.. autoclass:: IQuantizedConv3d
   :members:

.. autoclass:: IQuantizedConvTranspose1d
   :members:

.. autoclass:: IQuantizedConvTranspose2d
   :members:

.. autoclass:: IQuantizedConvTranspose3d
   :members:

.. autoclass:: IQuantizedRNN
   :members:

.. autoclass:: IQuantizedLSTM
   :members:

.. autoclass:: IQuantizedGRU
   :members:

.. autoclass:: IQuantizedMaxPool1d
   :members:

.. autoclass:: IQuantizedMaxPool2d
   :members:

.. autoclass:: IQuantizedMaxPool3d
   :members:

.. autoclass:: IQuantizedAvgPool1d
   :members:

.. autoclass:: IQuantizedAvgPool2d
   :members:

.. autoclass:: IQuantizedAvgPool3d
   :members:

.. autoclass:: IQuantizedAdaptiveAvgPool1d
   :members:

.. autoclass:: IQuantizedAdaptiveAvgPool2d
   :members:

.. autoclass:: IQuantizedAdaptiveAvgPool3d
   :members:

.. autoclass:: IQuantizedBatchNorm1d
   :members:

.. autoclass:: IQuantizedBatchNorm2d
   :members:

.. autoclass:: IQuantizedBatchNorm3d
   :members:

.. autoclass:: IQuantizedLayerNorm
   :members:

.. autoclass:: IQuantizedInstanceNorm1d
   :members:

.. autoclass:: IQuantizedInstanceNorm2d
   :members:

.. autoclass:: IQuantizedInstanceNorm3d
   :members:

.. autoclass:: IQuantizedGroupNorm
   :members:

.. autoclass:: IQuantizedMultiheadAttention
   :members:

.. autoclass:: IQuantizedEmbedding
   :members:

Integer Activation & Regularization Layers
------------------------------------------

.. autoclass:: IQuantizedReLU
   :members:

.. autoclass:: IQuantizedLeakyReLU
   :members:

.. autoclass:: IQuantizedSigmoid
   :members:

.. autoclass:: IQuantizedTanh
   :members:

.. autoclass:: IQuantizedGELU
   :members:

.. autoclass:: IQuantizedELU
   :members:

.. autoclass:: IQuantizedSiLU
   :members:

.. autoclass:: IQuantizedPReLU
   :members:

.. autoclass:: IQuantizedSoftmax
   :members:

.. autoclass:: IQuantizedDropout
   :members:

**Convenience aliases**

.. autodata:: IQuantizedAttention

.. autodata:: IQuantizedAvgPool

Usage Example
-------------

.. code-block:: python

    from pychop.layers import (
        ChopSTE, ChopfSTE, ChopiSTE,
        QuantizedConv2d, QuantizedReLU,
        IQuantizedLinear, IQuantizedReLU
    )

    # Floating-point QAT
    chop_fp = ChopSTE(exp_bits=5, sig_bits=10, rmode=3)

    # Fixed-point QAT
    chop_fixed = ChopfSTE(ibits=8, fbits=8, rmode=1)

    # Integer QAT
    chop_int = ChopiSTE(bits=8, symmetric=True)

    class MyQuantizedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = QuantizedConv2d(3, 64, 3, chop=chop_fp)
            self.relu  = QuantizedReLU(chop=chop_fp)
            self.fc    = IQuantizedLinear(512, 10, chop=chop_int)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

Post-Training Quantization (PTQ)
--------------------------------

.. code-block:: python

    from pychop.layers import post_quantization, ChopSTE

    chop = ChopSTE(exp_bits=8, sig_bits=23)   # or any other chop
    quantized_model = post_quantization(model, chop, eval_mode=True, verbose=True)