.. _bfp_formats:

====================================
Block Floating Point (BFP) Formats
====================================

Block Floating Point (BFP) is a quantization format where a group of numbers shares
a common exponent (scale factor), but each number has its own mantissa. This provides
a good balance between compression efficiency and hardware simplicity.


Overview
========

What is Block Floating Point?
------------------------------

Block Floating Point (BFP) divides data into blocks and applies a shared exponent
to all elements within each block. This is simpler than full floating-point but
provides better dynamic range than fixed-point quantization.

**Key Characteristics:**

- **Shared Exponent**: One exponent per block (typically 8 bits)
- **Individual Mantissas**: Each element has its own mantissa (4-16 bits)
- **Hardware-Efficient**: Simpler than full floating-point arithmetic
- **Good Dynamic Range**: Adapts to local data statistics

**BFP vs Other Formats:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Format
     - Memory
     - Dynamic Range
     - Hardware Cost
     - Best Use Case
   * - **BFP**
     - Low
     - Good
     - **Low**
     - **Edge devices, Inference**
   * - FP32
     - High
     - Excellent
     - High
     - Research, Training
   * - FP16
     - Medium
     - Good
     - Medium
     - Training, Inference
   * - INT8
     - Low
     - Poor
     - Low
     - Inference only
   * - MX Formats
     - Low
     - **Excellent**
     - Medium
     - Advanced training

Architecture
============

BFP Structure
-------------

A BFP block consists of:

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │         Block Floating Point Structure          │
   ├─────────────────────────────────────────────────┤
   │  Shared Exponent (8 bits)                       │
   ├─────────────────────────────────────────────────┤
   │  Element 1: Sign (1) + Mantissa (n bits)       │
   │  Element 2: Sign (1) + Mantissa (n bits)       │
   │  ...                                            │
   │  Element N: Sign (1) + Mantissa (n bits)       │
   └─────────────────────────────────────────────────┘

**Example: BFP8 with block_size=32**

- 1 shared exponent (8 bits)
- 32 elements × 8 bits each = 256 bits
- Total: 264 bits for 32 elements
- Compression vs FP16: 512/264 = **1.94x**

Predefined Formats
==================

Pychop provides several predefined BFP formats optimized for different use cases:

Standard Formats
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 20 20

   * - Format Name
     - Mantissa Bits
     - Block Size
     - Exponent Bits
     - Compression vs FP16
     - Use Case
   * - ``bfp16``
     - 16
     - 16
     - 8
     - 1.07x
     - High precision
   * - ``bfp12``
     - 12
     - 16
     - 8
     - 1.39x
     - Balanced
   * - ``bfp8``
     - 8
     - 32
     - 8
     - 1.94x
     - **Recommended default**
   * - ``bfp6``
     - 6
     - 32
     - 8
     - 2.56x
     - Aggressive compression
   * - ``bfp4``
     - 4
     - 32
     - 8
     - 3.76x
     - Ultra-low precision

Ultra-Low Precision Formats
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 20 20

   * - Format Name
     - Mantissa Bits
     - Block Size
     - Exponent Bits
     - Compression vs FP16
     - Use Case
   * - ``bfp3``
     - 3
     - 64
     - 8
     - 5.82x
     - Extreme compression
   * - ``bfp2``
     - 2
     - 128
     - 8
     - 10.67x
     - Research only

Intel Flexpoint Compatible
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 20 15

   * - Format Name
     - Mantissa Bits
     - Block Size
     - Exponent Bits
     - Compression vs FP16
     - Notes
   * - ``flexpoint16``
     - 16
     - 16
     - 5
     - 1.10x
     - Intel compatible
   * - ``flexpoint8``
     - 8
     - 32
     - 5
     - 1.97x
     - Intel compatible

Installation
============

BFP formats are included in Pychop. Ensure you have the required dependencies:

.. code-block:: bash

   # Basic installation (NumPy backend)
   pip install pychop

   # For PyTorch backend (recommended for training)
   pip install pychop torch

   # For JAX backend
   pip install pychop jax jaxlib flax

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   import pychop
   import numpy as np

   # Set backend (auto-detect by default)
   pychop.backend('auto')

   # Create test data
   X = np.random.randn(1024, 768).astype(np.float32)

   # Quantize with BFP8
   from pychop import bfp_quantize
   X_quantized = bfp_quantize(X, format='bfp8')

   # Check compression
   print(f"Original: {X.nbytes / 1024:.2f} KB")
   print(f"Quantized maintains same shape: {X_quantized.shape}")

Using BFPTensor
---------------

.. code-block:: python

   from pychop import BFPTensor

   # Create BFP tensor
   bfp = BFPTensor(X, format='bfp8')

   # Dequantize
   X_reconstructed = bfp.dequantize()

   # Get statistics
   stats = bfp.statistics()
   print(f"Compression: {stats['compression_ratio_fp16']:.2f}x vs FP16")
   print(f"Memory saved: {stats['memory_saved_vs_fp16']:.1f}%")

   # Compute error
   mse = np.mean((X - X_reconstructed) ** 2)
   print(f"MSE: {mse:.2e}")

Custom Formats
--------------

.. code-block:: python

   from pychop import create_bfp_spec, bfp_quantize

   # Create custom 5-bit BFP format
   custom_spec = create_bfp_spec(
       mantissa_bits=5,
       block_size=64,
       exponent_bits=8,
       name="my_bfp5"
   )

   # Use custom format
   X_q = bfp_quantize(X, format=custom_spec)

   # Or use tuple shorthand
   X_q = bfp_quantize(X, format=(5, 64))  # (mantissa_bits, block_size)

Backend-Specific Usage
======================

NumPy Backend
-------------

Pure NumPy implementation for inference and analysis:

.. code-block:: python

   import numpy as np
   import pychop

   pychop.backend('numpy')

   X = np.random.randn(512, 512).astype(np.float32)
   X_q = pychop.bfp_quantize(X, format='bfp8')

   # Compute reconstruction error
   error = np.mean((X - X_q) ** 2)
   print(f"MSE: {error:.2e}")

PyTorch Backend (with STE)
---------------------------

PyTorch backend with **Straight-Through Estimator** for Quantization-Aware Training:

.. code-block:: python

   import torch
   import pychop

   pychop.backend('torch')

   # Enable gradient tracking
   X = torch.randn(128, 768, requires_grad=True)

   # Quantize (automatic STE!)
   X_q = pychop.bfp_quantize(X, format='bfp8')

   # Backward pass - gradients flow through!
   loss = X_q.sum()
   loss.backward()

   print(f"Gradient shape: {X.grad.shape}")
   print(f"Gradient norm: {X.grad.norm():.2e}")

**Using BFP Quantizers in Models:**

.. code-block:: python

   from pychop.tch.bfp_formats import BFPQuantizerSTE

   class QuantizedModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.quantizer = BFPQuantizerSTE(format='bfp8')
           self.linear = torch.nn.Linear(768, 3072)
       
       def forward(self, x):
           x = self.quantizer(x)  # Quantize activations
           return self.linear(x)

   model = QuantizedModel()
   optimizer = torch.optim.Adam(model.parameters())

   # Training loop
   for batch in dataloader:
       output = model(batch)
       loss = loss_fn(output, target)
       loss.backward()  # STE handles gradients automatically!
       optimizer.step()

**Quantized Layers:**

.. code-block:: python

   from pychop.tch.bfp_formats import BFPLinear

   # Replace standard Linear with BFP quantized version
   layer = BFPLinear(
       in_features=768,
       out_features=3072,
       weight_format='bfp8',      # Quantize weights
       quantize_input=True,        # Quantize input activations
       quantize_output=False       # Keep output in FP32
   )

   x = torch.randn(32, 768)
   y = layer(x)  # Automatic quantization with STE

**Model Conversion:**

.. code-block:: python

   from pychop.tch.bfp_formats import convert_linear_to_bfp

   # Load pretrained model
   model = YourModel()

   # Convert all Linear layers to BFP
   model = convert_linear_to_bfp(
       model,
       format='bfp8',
       quantize_input=True,
       quantize_output=False,
       inplace=True
   )

   # Fine-tune with quantization
   for epoch in range(num_epochs):
       train(model)  # Gradients flow through STE automatically

JAX Backend (with Custom VJP)
------------------------------

JAX backend with custom Vector-Jacobian Product for differentiation:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import pychop

   pychop.backend('jax')

   # Create data
   key = jax.random.PRNGKey(0)
   X = jax.random.normal(key, (256, 512))

   # Quantize
   X_q = pychop.bfp_quantize(X, format='bfp8')

   # Test gradient flow
   from pychop.jx.bfp_formats import BFPQuantizerSTE

   quantizer = BFPQuantizerSTE(format='bfp8')

   def loss_fn(x):
       x_q = quantizer(x)
       return jnp.sum(x_q ** 2)

   # Compute gradients (custom VJP handles this)
   grad_fn = jax.grad(loss_fn)
   grads = grad_fn(X)

   print(f"Gradient shape: {grads.shape}")
   print(f"Gradient norm: {jnp.linalg.norm(grads):.2e}")

**Flax Integration:**

.. code-block:: python

   from flax import linen as nn
   from pychop.jx.bfp_formats import BFPDense

   class QuantizedMLP(nn.Module):
       features: list

       @nn.compact
       def __call__(self, x):
           for feat in self.features[:-1]:
               x = BFPDense(
                   features=feat,
                   weight_format='bfp8',
                   quantize_input=True
               )(x)
               x = nn.relu(x)
           
           x = BFPDense(features=self.features[-1])(x)
           return x

   model = QuantizedMLP(features=[512, 256, 128, 10])

   # Initialize
   key = jax.random.PRNGKey(0)
   x = jax.random.normal(key, (32, 784))
   variables = model.init(key, x)

   # Forward pass with quantization
   output = model.apply(variables, x)

API Reference
=============

Core Functions
--------------

bfp_quantize
~~~~~~~~~~~~

.. py:function:: bfp_quantize(data, format='bfp8', backend=None)

   Quantize array to BFP format with automatic backend detection.

   :param data: Input data (numpy.ndarray, torch.Tensor, or jax.Array)
   :type data: array-like
   :param format: BFP format specification
   :type format: str, BFPSpec, or tuple(int, int)
   :param backend: Force specific backend ('numpy', 'jax', or 'torch')
   :type backend: str, optional
   :return: Quantized data (same type as input)
   :rtype: array-like

   **Format Options:**

   - String: ``'bfp8'``, ``'bfp6'``, etc. (predefined formats)
   - Tuple: ``(mantissa_bits, block_size)`` for custom format
   - BFPSpec: Full specification object

   **Example:**

   .. code-block:: python

      import numpy as np
      from pychop import bfp_quantize

      X = np.random.randn(1024, 768)

      # Predefined format
      X_q = bfp_quantize(X, format='bfp8')

      # Custom format
      X_q = bfp_quantize(X, format=(6, 32))  # 6-bit mantissa, 32 elem/block

      # Force backend
      X_q = bfp_quantize(X, format='bfp8', backend='numpy')

Classes
-------

BFPTensor
~~~~~~~~~

.. py:class:: BFPTensor(data, format='bfp8', backend=None)

   Backend-agnostic BFP tensor wrapper.

   :param data: Input tensor
   :type data: array-like
   :param format: BFP format specification
   :type format: str, BFPSpec, or tuple
   :param backend: Force specific backend
   :type backend: str, optional

   **Methods:**

   .. py:method:: dequantize()

      Dequantize to original data type.

      :return: Reconstructed tensor
      :rtype: array-like

   .. py:method:: statistics()

      Get quantization statistics.

      :return: Dictionary with statistics
      :rtype: dict

      **Statistics Keys:**

      - ``format``: Format name
      - ``mantissa_bits``: Mantissa bits per element
      - ``block_size``: Elements per block
      - ``num_blocks``: Total number of blocks
      - ``compression_ratio_fp32``: Compression vs FP32
      - ``compression_ratio_fp16``: Compression vs FP16
      - ``bfp_memory_mb``: BFP memory usage (MB)
      - ``memory_saved_vs_fp16``: Memory saved vs FP16 (%)
      - ``bits_per_element``: Average bits per element

   **Example:**

   .. code-block:: python

      from pychop import BFPTensor

      bfp = BFPTensor(X, format='bfp8')

      # Reconstruct
      X_reconstructed = bfp.dequantize()

      # Get statistics
      stats = bfp.statistics()
      print(f"Compression: {stats['compression_ratio_fp16']:.2f}x")
      print(f"Memory saved: {stats['memory_saved_vs_fp16']:.1f}%")
      print(f"Blocks: {stats['num_blocks']}")

BFPSpec
~~~~~~~

.. py:class:: BFPSpec(name, mantissa_bits, block_size, exponent_bits=8, has_sign=True, use_subnormals=False)

   BFP format specification.

   :param name: Format name
   :type name: str
   :param mantissa_bits: Mantissa bits per element
   :type mantissa_bits: int
   :param block_size: Elements per block
   :type block_size: int
   :param exponent_bits: Shared exponent bits
   :type exponent_bits: int
   :param has_sign: Whether elements have sign bits
   :type has_sign: bool
   :param use_subnormals: Whether to support subnormal numbers
   :type use_subnormals: bool

   **Properties:**

   - ``total_bits_per_block``: Total bits for entire block
   - ``compression_vs_fp32``: Compression ratio vs FP32
   - ``compression_vs_fp16``: Compression ratio vs FP16

create_bfp_spec
~~~~~~~~~~~~~~~

.. py:function:: create_bfp_spec(mantissa_bits, block_size, exponent_bits=8, name=None)

   Create custom BFP format specification.

   :param mantissa_bits: Number of mantissa bits (1-32)
   :type mantissa_bits: int
   :param block_size: Elements per block
   :type block_size: int
   :param exponent_bits: Bits for shared exponent
   :type exponent_bits: int
   :param name: Custom format name
   :type name: str, optional
   :return: BFP format specification
   :rtype: BFPSpec

   **Example:**

   .. code-block:: python

      from pychop import create_bfp_spec, bfp_quantize

      # Create 5-bit BFP format
      spec = create_bfp_spec(
          mantissa_bits=5,
          block_size=64,
          exponent_bits=8,
          name="my_bfp5"
      )

      # Use custom format
      X_q = bfp_quantize(X, format=spec)

Utility Functions
-----------------

print_bfp_format_table
~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: print_bfp_format_table()

   Print table of all predefined BFP formats.

   **Example:**

   .. code-block:: python

      from pychop import print_bfp_format_table

      print_bfp_format_table()

   **Output:**

   .. code-block:: text

      ==========================================================================================
      Predefined BFP Formats
      ==========================================================================================
      Name            Mantissa   Block Size   Exponent   Compress FP16   Total Bits
      ------------------------------------------------------------------------------------------
      bfp16           16         16           8          1.07x            264
      bfp12           12         16           8          1.39x            200
      bfp8            8          32           8          1.94x            264
      bfp6            6          32           8          2.56x            200
      bfp4            4          32           8          3.76x            136
      bfp3            3          64           8          5.82x            200
      bfp2            2          128          8          10.67x           264
      flexpoint16     16         16           5          1.10x            261
      flexpoint8      8          32           5          1.97x            261
      ==========================================================================================

PyTorch-Specific API
--------------------

BFPQuantizerSTE
~~~~~~~~~~~~~~~

.. py:class:: pychop.tch.bfp_formats.BFPQuantizerSTE(format='bfp8')

   BFP quantizer with Straight-Through Estimator for QAT.

   Automatically uses STE during training (``requires_grad=True``).

   :param format: BFP format specification
   :type format: str, BFPSpec, or tuple

   **Example:**

   .. code-block:: python

      import torch
      from pychop.tch.bfp_formats import BFPQuantizerSTE

      quantizer = BFPQuantizerSTE(format='bfp8')

      x = torch.randn(32, 768, requires_grad=True)
      x_q = quantizer(x)

      loss = x_q.sum()
      loss.backward()  # Gradients flow through STE

BFPLinear
~~~~~~~~~

.. py:class:: pychop.tch.bfp_formats.BFPLinear(in_features, out_features, bias=True, weight_format='bfp8', act_format=None, quantize_input=True, quantize_output=False)

   Linear layer with BFP quantization.

   :param in_features: Input dimension
   :type in_features: int
   :param out_features: Output dimension
   :type out_features: int
   :param bias: Whether to use bias
   :type bias: bool
   :param weight_format: BFP format for weights
   :type weight_format: str, BFPSpec, or tuple
   :param act_format: BFP format for activations (if None, uses weight_format)
   :type act_format: str, BFPSpec, or tuple, optional
   :param quantize_input: Whether to quantize input
   :type quantize_input: bool
   :param quantize_output: Whether to quantize output
   :type quantize_output: bool

   **Example:**

   .. code-block:: python

      from pychop.tch.bfp_formats import BFPLinear

      layer = BFPLinear(
          in_features=768,
          out_features=3072,
          weight_format='bfp8',
          quantize_input=True,
          quantize_output=False
      )

      x = torch.randn(32, 768)
      y = layer(x)  # Automatic quantization with STE

BFPConv2d
~~~~~~~~~

.. py:class:: pychop.tch.bfp_formats.BFPConv2d(in_channels, out_channels, kernel_size, weight_format='bfp8', act_format=None, quantize_input=True, quantize_output=False, **kwargs)

   2D Convolution with BFP quantization.

   :param in_channels: Input channels
   :type in_channels: int
   :param out_channels: Output channels
   :type out_channels: int
   :param kernel_size: Convolution kernel size
   :type kernel_size: int or tuple
   :param weight_format: BFP format for weights
   :type weight_format: str, BFPSpec, or tuple
   :param act_format: BFP format for activations
   :type act_format: str, BFPSpec, or tuple, optional
   :param quantize_input: Whether to quantize input
   :type quantize_input: bool
   :param quantize_output: Whether to quantize output
   :type quantize_output: bool
   :param kwargs: Other Conv2d parameters
   :type kwargs: dict

   **Example:**

   .. code-block:: python

      from pychop.tch.bfp_formats import BFPConv2d

      conv = BFPConv2d(
          in_channels=3,
          out_channels=64,
          kernel_size=3,
          weight_format='bfp8',
          quantize_input=True,
          padding=1
      )

      x = torch.randn(16, 3, 224, 224)
      y = conv(x)

convert_linear_to_bfp
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: pychop.tch.bfp_formats.convert_linear_to_bfp(module, format='bfp8', quantize_input=True, quantize_output=False, inplace=True)

   Convert all Linear layers in a model to BFP quantized versions.

   :param module: Model to convert
   :type module: torch.nn.Module
   :param format: BFP format
   :type format: str, BFPSpec, or tuple
   :param quantize_input: Whether to quantize inputs
   :type quantize_input: bool
   :param quantize_output: Whether to quantize outputs
   :type quantize_output: bool
   :param inplace: Whether to modify in place
   :type inplace: bool
   :return: Converted model
   :rtype: torch.nn.Module

   **Example:**

   .. code-block:: python

      from pychop.tch.bfp_formats import convert_linear_to_bfp
      import transformers

      # Load pretrained model
      model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

      # Convert to BFP8
      model = convert_linear_to_bfp(
          model,
          format='bfp8',
          quantize_input=True,
          quantize_output=False,
          inplace=True
      )

      # Fine-tune with BFP quantization
      for epoch in range(num_epochs):
          train(model)

JAX-Specific API
----------------

BFPQuantizerSTE (JAX)
~~~~~~~~~~~~~~~~~~~~~

.. py:class:: pychop.jx.bfp_formats.BFPQuantizerSTE(format='bfp8')

   BFP quantizer with custom VJP for JAX.

   :param format: BFP format specification
   :type format: str, BFPSpec, or tuple

   **Example:**

   .. code-block:: python

      import jax.numpy as jnp
      from pychop.jx.bfp_formats import BFPQuantizerSTE

      quantizer = BFPQuantizerSTE(format='bfp8')

      x = jnp.array(np.random.randn(256, 512))
      x_q = quantizer(x)

BFPDense
~~~~~~~~

.. py:class:: pychop.jx.bfp_formats.BFPDense(features, use_bias=True, weight_format='bfp8', quantize_input=True)

   Dense layer with BFP quantization for Flax.

   :param features: Number of output features
   :type features: int
   :param use_bias: Whether to use bias
   :type use_bias: bool
   :param weight_format: BFP format for weights
   :type weight_format: str, BFPSpec, or tuple
   :param quantize_input: Whether to quantize input
   :type quantize_input: bool

   **Example:**

   .. code-block:: python

      from flax import linen as nn
      from pychop.jx.bfp_formats import BFPDense

      class MyModel(nn.Module):
          @nn.compact
          def __call__(self, x):
              x = BFPDense(features=512, weight_format='bfp8')(x)
              x = nn.relu(x)
              x = BFPDense(features=10)(x)
              return x

Advanced Usage
==============

Format Comparison
-----------------

Compare different BFP formats on the same data:

.. code-block:: python

   import numpy as np
   from pychop import BFPTensor

   X = np.random.randn(1024, 768).astype(np.float32)

   formats = ['bfp16', 'bfp8', 'bfp6', 'bfp4']

   print("Format Comparison")
   print("="*80)
   print(f"{'Format':<10} {'Compression':<15} {'MSE':<12} {'MAE':<12}")
   print("-"*80)

   for fmt in formats:
       bfp = BFPTensor(X, format=fmt)
       X_reconstructed = bfp.dequantize()
       stats = bfp.statistics()
       
       mse = np.mean((X - X_reconstructed) ** 2)
       mae = np.mean(np.abs(X - X_reconstructed))
       
       print(f"{fmt:<10} {stats['compression_ratio_fp16']:.2f}x{'':>11} "
             f"{mse:.2e}{'':>6} {mae:.2e}")

Memory Analysis
---------------

Analyze memory usage for different formats:

.. code-block:: python

   from pychop import BFPTensor

   X = np.random.randn(4096, 4096).astype(np.float32)

   print("\nMemory Analysis")
   print("="*80)
   print(f"Original FP32: {X.nbytes / 1024**2:.2f} MB")
   print(f"FP16 equivalent: {X.nbytes / 2 / 1024**2:.2f} MB")
   print("-"*80)

   for fmt in ['bfp8', 'bfp6', 'bfp4']:
       bfp = BFPTensor(X, format=fmt)
       stats = bfp.statistics()
       
       print(f"\n{fmt.upper()}:")
       print(f"  Memory: {stats['bfp_memory_mb']:.2f} MB")
       print(f"  Saved vs FP32: {stats['memory_saved_vs_fp32']:.1f}%")
       print(f"  Saved vs FP16: {stats['memory_saved_vs_fp16']:.1f}%")
       print(f"  Compression: {stats['compression_ratio_fp16']:.2f}x vs FP16")

LLM Fine-Tuning Example
-----------------------

Complete example for fine-tuning LLMs with BFP quantization:

.. code-block:: python

   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from pychop.tch.bfp_formats import convert_linear_to_bfp

   # Load model
   model = AutoModelForCausalLM.from_pretrained("gpt2")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")

   # Convert to BFP8
   model = convert_linear_to_bfp(
       model,
       format='bfp8',
       quantize_input=True,
       quantize_output=False,
       inplace=True
   )

   # Setup training
   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = model.to(device)

   # Training loop
   model.train()
   for epoch in range(num_epochs):
       for batch in dataloader:
           input_ids = batch['input_ids'].to(device)
           labels = input_ids.clone()
           
           # Forward pass (automatic BFP quantization with STE)
           outputs = model(input_ids=input_ids, labels=labels)
           loss = outputs.loss
           
           # Backward pass (gradients flow through STE)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           print(f"Loss: {loss.item():.4f}")

   # Save quantized model
   torch.save(model.state_dict(), 'model_bfp8.pt')

Performance Tips
================

Choosing Block Size
-------------------

Block size affects compression and accuracy:

- **Small blocks (8-16)**: Better accuracy, less compression
- **Medium blocks (32)**: **Recommended default**, good balance
- **Large blocks (64-128)**: Higher compression, lower accuracy

.. code-block:: python

   # Test different block sizes
   for block_size in [8, 16, 32, 64, 128]:
       X_q = bfp_quantize(X, format=(8, block_size))
       mse = np.mean((X - X_q) ** 2)
       print(f"Block size {block_size}: MSE = {mse:.2e}")

Choosing Mantissa Bits
-----------------------

Mantissa bits control precision:

- **16 bits**: Near-lossless, minimal compression
- **8 bits**: **Recommended for most tasks**
- **6 bits**: Aggressive compression, acceptable for inference
- **4 bits or less**: Research/experimental

Backend Selection
-----------------

Choose backend based on your needs:

.. code-block:: python

   # For inference (fastest)
   pychop.backend('numpy')

   # For training (STE support)
   pychop.backend('torch')

   # For JAX/Flax (custom VJP)
   pychop.backend('jax')

   # Auto-detect (recommended)
   pychop.backend('auto')

Troubleshooting
===============

Common Issues
-------------

**Import Error:**

.. code-block:: python

   # Error: cannot import name 'bfp_quantize'
   # Solution: Update pychop
   pip install --upgrade pychop

**Memory Issues:**

.. code-block:: python

   # For very large tensors, use smaller block sizes
   X_q = bfp_quantize(X, format=(8, 16))  # Smaller blocks

**Gradient Issues:**

.. code-block:: python

   # Ensure requires_grad=True for training
   X = torch.randn(128, 768, requires_grad=True)
   X_q = bfp_quantize(X, format='bfp8')
   
   # Check gradient flow
   loss = X_q.sum()
   loss.backward()
   assert X.grad is not None, "Gradients not flowing!"

**Backend Issues:**

.. code-block:: python

   # Check current backend
   import pychop
   print(pychop.get_backend())
   
   # Reset backend
   pychop.backend('auto')

FAQ
===

**Q: What's the difference between BFP and MX formats?**

A: BFP uses one shared exponent per block, while MX formats use both a shared scale
and individual exponents per element. BFP is simpler and more hardware-efficient,
while MX provides better dynamic range.

**Q: Can I use BFP for training?**

A: Yes! The PyTorch backend includes Straight-Through Estimator (STE) support,
enabling full quantization-aware training. JAX backend uses custom VJP.

**Q: Which format should I use?**

A: For most cases, **BFP8** (8-bit mantissa, 32 elements/block) is recommended.
It provides ~2x compression vs FP16 with minimal accuracy loss.

**Q: How does BFP compare to INT8?**

A: BFP provides better dynamic range than INT8 while maintaining similar compression.
BFP adapts to local data statistics (per-block), while INT8 uses global scaling.

**Q: Can I mix different formats in the same model?**

A: Yes! You can use different formats for different layers:

.. code-block:: python

   from pychop.tch.bfp_formats import BFPLinear

   class MixedPrecisionModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Higher precision for first layer
           self.fc1 = BFPLinear(768, 3072, weight_format='bfp12')
           # Lower precision for middle layers
           self.fc2 = BFPLinear(3072, 3072, weight_format='bfp6')
           # Full precision for output
           self.fc3 = nn.Linear(3072, 768)

**Q: Does BFP work with quantized models from PyTorch/TensorFlow?**

A: BFP is independent of PyTorch/TensorFlow quantization. You can apply BFP
quantization to any model, including already-quantized models.

References
==========

**Papers:**

1. Intel Flexpoint: "Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks" (2017)
   https://arxiv.org/abs/1711.02213

2. Microsoft BFloat16: "BFloat16: The Secret to High Performance on Cloud TPUs" (2019)
   https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus

3. Block Floating Point for Neural Networks: "Training Deep Neural Networks with 8-bit Floating Point Numbers" (2018)
   https://arxiv.org/abs/1812.08011

**Related Formats:**

- :ref:`mx_formats` - OCP Microscaling formats with better dynamic range
- :ref:`fixed_point` - Fixed-point quantization (Chopf)
- :ref:`integer` - Integer quantization (Chopi)

**External Links:**

- `Pychop GitHub <https://github.com/inEXASCALE/pychop>`_
- `OCP MX Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_
- `Intel Flexpoint Paper <https://arxiv.org/abs/1711.02213>`_

.. note::
   For the latest updates and examples, see the `Pychop GitHub repository <https://github.com/inEXASCALE/pychop>`_.