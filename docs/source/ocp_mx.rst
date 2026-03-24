.. _mx_formats:

Microscaling (MX) formats
==========================

.. currentmodule:: pychop.mx_formats

Overview
--------

`Microscaling (MX) formats <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_  are a revolutionary approach to low-precision arithmetic that uses
**block-level shared exponents** to achieve extreme compression while maintaining reasonable
accuracy. Developed by the Open Compute Project (OCP), MX formats are particularly effective
for deep learning workloads.

**Key Innovation**: Instead of each number having its own exponent, a group of numbers (a "block")
shares a single scale factor, allowing individual elements to use very few bits (even 2-3 bits!)
while maintaining a large dynamic range.

.. code-block:: text

    Traditional Floating Point:
    [S E E E M M M] [S E E E M M M] [S E E E M M M] ...
     ↑ Each has own exponent

    Microscaling Format:
    [Scale Factor: 8 bits] [M M M] [M M M] [M M M] [M M M] ...
     ↑ Shared exponent      ↑ Only 3-4 bits per element!


Architecture
------------

Block Structure
~~~~~~~~~~~~~~~

Each MX block consists of:

1. **Shared Scale Factor** (typically 8 bits, E8M0 format)
   
   - A single floating-point number that scales all elements in the block
   - Usually represents a power of 2: ``2^scale``
   - Default format: E8M0 (8-bit exponent, 0-bit mantissa)
   - Can be customized: E6M0, E10M0, E8M1, etc.

2. **Data Elements** (typically 4-8 bits each)
   
   - Multiple low-precision floating-point numbers
   - Each element is multiplied by the shared scale factor
   - Common formats: E4M3 (8-bit), E2M1 (4-bit), E3M2 (6-bit)
   - **Fully customizable**: any E/M combination supported

**Reconstruction Formula**:

.. math::

    \text{actual\_value}_i = \text{element}_i \times 2^{\text{scale\_factor}}



This design offers significantly better dynamic range than Block Floating Point (BFP) while maintaining hardware efficiency.

.. note::
   **MX vs BFP:**
   
   - **BFP**: All elements share ONE exponent → simpler, less dynamic range
   - **MX**: Shared scale + individual exponents → better range, more flexible
   - **Use MX** when you need maximum accuracy with compression
   - **Use BFP** when you need maximum simplicity for edge devices

Key Features
---------------- 

 **OCP Standard Compliance**
   - Full support for MXFP8, MXFP6, MXFP4, and MXINT8 formats
   - Compatible with OCP Microscaling Formats v1.0 spec

 **Multi-Backend Support**
   - **NumPy**: Pure numerical computation (inference, analysis)
   - **PyTorch**: Straight-Through Estimator (STE) for QAT
   - **JAX**: Custom VJP for differentiation

 **Automatic Backend Detection**
   - Automatically detects input type (numpy.ndarray, torch.Tensor, jax.Array)
   - No manual backend switching needed

 **Flexible Quantization**
   - Predefined OCP formats
   - Custom format creation (exp_bits, sig_bits)
   - Configurable block sizes

MX Format Specification
-------------------------------- 

Structure
---------

Each MX block contains:

.. code-block:: text

   MX Block:
   ┌──────────────────────────────────────────────────┐
   │ Shared Scale (8 bits)                            │
   ├──────────────────────────────────────────────────┤
   │ Element 1: [sign(1) | exp(e) | mantissa(m)]     │
   │ Element 2: [sign(1) | exp(e) | mantissa(m)]     │
   │ ...                                              │
   │ Element N: [sign(1) | exp(e) | mantissa(m)]     │
   └──────────────────────────────────────────────────┘

   Total bits per block = scale_bits + (1+e+m) × block_size

Predefined Formats
------------------

Pychop supports all OCP standard MX formats:

.. list-table:: OCP Microscaling (MX) Formats
   :header-rows: 1
   :widths: 20 15 10 10 15 15 15

   * - Format
     - Element
     - Block
     - Scale
     - Bits/Elem
     - Compress FP16
     - Use Case
   * - MXFP8_E5M2
     - E5M2 (8-bit)
     - 32
     - 8b
     - 8.25
     - 1.94×
     - Wide range
   * - MXFP8_E4M3
     - E4M3 (8-bit)
     - 32
     - 8b
     - 8.25
     - 1.94×
     - Balanced
   * - MXFP6_E3M2
     - E3M2 (6-bit)
     - 32
     - 8b
     - 6.25
     - 2.56×
     - High compression
   * - MXFP6_E2M3
     - E2M3 (6-bit)
     - 32
     - 8b
     - 6.25
     - 2.56×
     - More precision
   * - MXFP4_E2M1
     - E2M1 (4-bit)
     - 32
     - 8b
     - 4.25
     - 3.76×
     - Extreme compression
   * - MXINT8
     - E0M7 (8-bit)
     - 32
     - 8b
     - 8.25
     - 1.94×
     - Integer-like

Quick Start
-----------


NumPy Backend
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pychop import mx_quantize, MXTensor
   
   # Set backend (optional, auto-detected)
   import pychop
   pychop.backend('numpy')
   
   # Create data
   X = np.random.randn(1024, 768).astype(np.float32)
   
   # Quantize with MXFP8 E4M3
   X_q = mx_quantize(X, format='mxfp8_e4m3')
   
   # Compute error
   mse = np.mean((X - X_q) ** 2)
   print(f"MSE: {mse:.2e}")
   
   # Get statistics
   mx = MXTensor(X, format='mxfp8_e4m3')
   stats = mx.statistics()
   print(f"Compression: {stats['compression_ratio_fp16']:.2f}x vs FP16")

PyTorch Backend (with STE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from pychop import mx_quantize
   
   pychop.backend('torch')
   
   # Create data with gradient
   X = torch.randn(128, 768, requires_grad=True)
   
   # Quantize (automatic STE!)
   X_q = mx_quantize(X, format='mxfp8_e4m3')
   
   # Backward pass - gradients flow through!
   loss = X_q.sum()
   loss.backward()
   
   print(f"Gradient shape: {X.grad.shape}")
   print(f"Gradient norm: {X.grad.norm():.2e}")

JAX Backend (with Custom VJP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from pychop import mx_quantize
   
   pychop.backend('jax')
   
   # Create data
   key = jax.random.PRNGKey(0)
   X = jax.random.normal(key, (256, 512))
   
   # Quantize
   X_q = mx_quantize(X, format='mxfp8_e4m3')
   
   # Test gradient flow
   def loss_fn(x):
       x_q = mx_quantize(x, format='mxfp8_e4m3')
       return jnp.sum(x_q ** 2)
   
   grad_fn = jax.grad(loss_fn)
   grads = grad_fn(X)
   print(f"Gradient norm: {jnp.linalg.norm(grads):.2e}")

API Reference
---------------- 


mx_quantize
~~~~~~~~~~~

.. py:function:: mx_quantize(data, format='mxfp8_e4m3', block_size=32, scale_exp_bits=None, scale_sig_bits=None, backend=None)

   Quantize array to MX format.
   
   Automatically detects backend from input type or uses specified backend.
   
   :param data: Input data (numpy.ndarray, torch.Tensor, or jax.Array)
   :type data: array-like
   :param format: MX format specification (string, MXSpec, or tuple)
   :type format: str, MXSpec, or tuple
   :param block_size: Number of elements per block
   :type block_size: int
   :param scale_exp_bits: Override scale exponent bits (optional)
   :type scale_exp_bits: int or None
   :param scale_sig_bits: Override scale significand bits (optional)
   :type scale_sig_bits: int or None
   :param backend: Force specific backend ('numpy', 'jax', or 'torch')
   :type backend: str or None
   :return: Quantized data (same type as input)
   :rtype: array-like
   
   **Examples:**
   
   .. code-block:: python
   
      # NumPy - standard format
      X_q = mx_quantize(X_np, format='mxfp8_e4m3')
      
      # PyTorch - with automatic STE
      X_q = mx_quantize(X_torch, format='mxfp8_e4m3')
      loss = X_q.sum()
      loss.backward()  # Gradients flow through!
      
      # Custom format (4-bit exponent, 3-bit mantissa)
      X_q = mx_quantize(X, format=(4, 3), block_size=32)
      
      # Override scale bits
      X_q = mx_quantize(X, format='mxfp8_e4m3', scale_exp_bits=5)

MXTensor
~~~~~~~~

.. py:class:: MXTensor(data, format='mxfp8_e4m3', block_size=32, scale_exp_bits=None, scale_sig_bits=None, backend=None)

   Backend-agnostic MX tensor wrapper.
   
   Automatically routes to appropriate backend implementation.
   
   :param data: Input tensor
   :type data: array-like
   :param format: MX format specification
   :type format: str, MXSpec, or tuple
   :param block_size: Elements per block
   :type block_size: int
   :param scale_exp_bits: Override scale exponent bits
   :type scale_exp_bits: int or None
   :param scale_sig_bits: Override scale significand bits
   :type scale_sig_bits: int or None
   :param backend: Force specific backend
   :type backend: str or None
   
   .. py:method:: dequantize()
   
      Dequantize to original data type.
      
      :return: Reconstructed data
      :rtype: array-like
   
   .. py:method:: statistics()
   
      Get quantization statistics.
      
      :return: Dictionary with compression ratios, errors, etc.
      :rtype: dict
   
   **Example:**
   
   .. code-block:: python
   
      import numpy as np
      from pychop import MXTensor
      
      X = np.random.randn(1024, 768)
      mx = MXTensor(X, format='mxfp8_e4m3')
      
      # Dequantize
      X_reconstructed = mx.dequantize()
      
      # Get statistics
      stats = mx.statistics()
      print(f"Format: {stats['format']}")
      print(f"Blocks: {stats['num_blocks']}")
      print(f"Compression: {stats['compression_ratio_fp16']:.2f}x")

MXSpec
~~~~~~

.. py:class:: MXSpec(name, exp_bits, sig_bits, block_size=32, scale_exp_bits=8, scale_sig_bits=0)

   MX format specification.
   
   :param name: Format name
   :type name: str
   :param exp_bits: Element exponent bits
   :type exp_bits: int
   :param sig_bits: Element significand bits (excluding implicit 1)
   :type sig_bits: int
   :param block_size: Elements per block
   :type block_size: int
   :param scale_exp_bits: Scale factor exponent bits
   :type scale_exp_bits: int
   :param scale_sig_bits: Scale factor significand bits
   :type scale_sig_bits: int
   
   .. py:attribute:: element_bits
   
      Total bits per element (1 sign + exp + sig).
      
      :type: int
   
   .. py:attribute:: total_bits_per_block
   
      Total bits for entire block (elements + scale).
      
      :type: int
   
   .. py:attribute:: compression_vs_fp16
   
      Compression ratio compared to FP16.
      
      :type: float

Utility Functions
-----------------

create_mx_spec
~~~~~~~~~~~~~~

.. py:function:: create_mx_spec(exp_bits, sig_bits, block_size=32, scale_exp_bits=8, name=None)

   Create custom MX format specification.
   
   :param exp_bits: Element exponent bits
   :type exp_bits: int
   :param sig_bits: Element significand bits
   :type sig_bits: int
   :param block_size: Elements per block
   :type block_size: int
   :param scale_exp_bits: Scale exponent bits
   :type scale_exp_bits: int
   :param name: Custom format name (auto-generated if None)
   :type name: str or None
   :return: MX format specification
   :rtype: MXSpec
   
   **Example:**
   
   .. code-block:: python
   
      from pychop import create_mx_spec
      
      # Custom 5-bit format (E3M1)
      spec = create_mx_spec(exp_bits=3, sig_bits=1, block_size=32)
      print(spec.name)  # "CUSTOM_MX5_E3M1"
      print(spec.compression_vs_fp16)  # 3.2x

compare_mx_formats
~~~~~~~~~~~~~~~~~~

.. py:function:: compare_mx_formats(data, formats=None, block_size=32)

   Compare different MX formats on the same data.
   
   :param data: Test data
   :type data: array-like
   :param formats: List of format names to compare (uses all if None)
   :type formats: list or None
   :param block_size: Elements per block
   :type block_size: int
   
   **Example:**
   
   .. code-block:: python
   
      import numpy as np
      from pychop import compare_mx_formats
      
      X = np.random.randn(1024, 512).astype(np.float32)
      
      # Compare all formats
      compare_mx_formats(X)
      
      # Compare specific formats
      compare_mx_formats(X, formats=['mxfp8_e4m3', 'mxfp6_e3m2', 'mxfp4_e2m1'])

print_mx_format_table
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: print_mx_format_table()

   Print table of all predefined MX formats.
   
   **Example:**
   
   .. code-block:: python
   
      from pychop import print_mx_format_table
      
      print_mx_format_table()

PyTorch Backend (QAT)
-------------------------------- 

For **Quantization-Aware Training** in PyTorch, use the ``tch`` submodule:

Quantizer with STE
------------------

.. code-block:: python

   from pychop.tch.mx_formats import MXQuantizerSTE
   
   # Create quantizer
   quantizer = MXQuantizerSTE(format='mxfp8_e4m3', block_size=32)
   
   # Use in forward pass
   x = torch.randn(128, 768, requires_grad=True)
   x_q = quantizer(x)  # Automatic STE!
   
   # Backward pass
   loss = x_q.sum()
   loss.backward()  # Gradients flow through

Quantized Linear Layer
----------------------

.. code-block:: python

   from pychop.tch.mx_formats import MXLinear
   
   # Create quantized linear layer
   layer = MXLinear(
       in_features=768,
       out_features=3072,
       bias=True,
       weight_format='mxfp8_e4m3',
       act_format='mxfp8_e4m3',  # Optional: quantize activations
       block_size=32,
       quantize_input=True,
       quantize_output=False
   )
   
   # Use like normal Linear layer
   x = torch.randn(32, 768, requires_grad=True)
   y = layer(x)
   
   # Backward pass works automatically
   loss = y.sum()
   loss.backward()

Quantized Attention
-------------------

.. code-block:: python

   from pychop.tch.mx_formats import MXAttention
   
   # Multi-head attention with MX quantization
   attn = MXAttention(
       embed_dim=768,
       num_heads=12,
       format='mxfp8_e4m3',
       block_size=32,
       dropout=0.1
   )
   
   # Forward pass
   query = torch.randn(16, 128, 768)  # [batch, seq_len, embed_dim]
   output = attn(query)
   
   # For cross-attention
   key = torch.randn(16, 64, 768)
   value = torch.randn(16, 64, 768)
   output = attn(query, key, value)

Model Conversion
----------------

Convert existing PyTorch models to use MX quantization:

.. code-block:: python

   from pychop.tch.mx_formats import convert_linear_to_mx
   import torch.nn as nn
   
   # Original model
   class OriginalModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(768, 3072)
           self.fc2 = nn.Linear(3072, 768)
       
       def forward(self, x):
           return self.fc2(torch.relu(self.fc1(x)))
   
   model = OriginalModel()
   
   # Convert all Linear layers to MX quantized
   model_mx = convert_linear_to_mx(
       model,
       format='mxfp8_e4m3',
       block_size=32,
       quantize_input=True,
       quantize_output=False,
       inplace=True  # Modify in-place
   )
   
   # Now train with MX quantization!
   optimizer = torch.optim.Adam(model_mx.parameters(), lr=1e-4)
   
   for batch in dataloader:
       optimizer.zero_grad()
       output = model_mx(batch)
       loss = criterion(output, target)
       loss.backward()  # Gradients flow through STE
       optimizer.step()

LLM Fine-tuning
---------------

Example: Fine-tune Transformer with MX quantization:

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   from pychop.tch.mx_formats import convert_linear_to_mx
   import torch
   
   # Load model
   model = AutoModelForCausalLM.from_pretrained("gpt2")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   
   # Convert to MX quantization
   model = convert_linear_to_mx(
       model,
       format='mxfp8_e4m3',  # 1.94x compression vs FP16
       block_size=32,
       quantize_input=True,
       inplace=True
   )
   
   # Move to GPU
   device = 'cuda'
   model = model.to(device)
   
   # Training loop
   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
   
   for epoch in range(num_epochs):
       for batch in dataloader:
           input_ids = batch['input_ids'].to(device)
           labels = input_ids.clone()
           
           # Forward pass (with MX quantization)
           outputs = model(input_ids=input_ids, labels=labels)
           loss = outputs.loss
           
           # Backward pass (STE automatic)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           print(f"Loss: {loss.item():.4f}")

JAX Backend
------------

For JAX/Flax training, use the ``jx`` submodule:

Quantizer with Custom VJP
--------------------------

.. code-block:: python

   from pychop.jx.mx_formats import MXQuantizerSTE
   import jax
   import jax.numpy as jnp
   
   # Create quantizer
   quantizer = MXQuantizerSTE(format='mxfp8_e4m3', block_size=32)
   
   # Define loss function
   def loss_fn(x):
       x_q = quantizer(x)
       return jnp.sum(x_q ** 2)
   
   # Gradient flows through custom VJP
   grad_fn = jax.grad(loss_fn)
   
   x = jax.random.normal(jax.random.PRNGKey(0), (128, 768))
   grads = grad_fn(x)

Quantized Dense Layer (Flax)
-----------------------------

.. code-block:: python

   from pychop.jx.mx_formats import MXDense
   from flax import linen as nn
   import jax.numpy as jnp
   
   class QuantizedMLP(nn.Module):
       num_classes: int = 10
       
       @nn.compact
       def __call__(self, x):
           # MX quantized dense layers
           x = MXDense(
               features=256,
               weight_format='mxfp8_e4m3',
               quantize_input=True
           )(x)
           x = nn.relu(x)
           
           x = MXDense(
               features=self.num_classes,
               weight_format='mxfp8_e4m3',
               quantize_input=True
           )(x)
           
           return x
   
   # Initialize and train
   model = QuantizedMLP()
   key = jax.random.PRNGKey(0)
   x = jax.random.normal(key, (32, 784))
   variables = model.init(key, x)
   
   # Forward pass
   output = model.apply(variables, x)

Advanced Usage
---------------- 

Create custom MX formats for specific use cases:

.. code-block:: python

   from pychop import create_mx_spec, mx_quantize
   
   # Ultra-low precision format (3-bit total)
   # 1 sign bit + 1 exp bit + 1 mantissa bit
   ultra_low = create_mx_spec(
       exp_bits=1,
       sig_bits=1,
       block_size=64,  # Larger blocks for better amortization
       scale_exp_bits=8,
       name="MXFP3_E1M1"
   )
   
   print(f"Compression: {ultra_low.compression_vs_fp16:.2f}x")
   # Output: Compression: 5.33x
   
   # Use the custom format
   X_q = mx_quantize(X, format=ultra_low)
   
   # Or use tuple shorthand
   X_q = mx_quantize(X, format=(1, 1), block_size=64)

Fine-grained Control
--------------------

Override scale parameters for advanced control:

.. code-block:: python

   from pychop import mx_quantize
   
   # Use smaller scale exponent for better precision
   X_q = mx_quantize(
       X,
       format='mxfp8_e4m3',
       block_size=32,
       scale_exp_bits=5,  # Override default 8-bit scale
       scale_sig_bits=0
   )

Block Size Selection
--------------------

Choose block size based on data characteristics:

.. code-block:: python

   import numpy as np
   from pychop import mx_quantize
   
   # For uniform data: larger blocks
   X_uniform = np.random.randn(1024, 768)
   X_q = mx_quantize(X_uniform, format='mxfp8_e4m3', block_size=64)
   
   # For varying data: smaller blocks
   X_varying = np.concatenate([
       np.random.randn(512, 768) * 0.1,  # Small values
       np.random.randn(512, 768) * 10.0  # Large values
   ])
   X_q = mx_quantize(X_varying, format='mxfp8_e4m3', block_size=16)

Performance Tips
-----------------


Memory Reduction
~~~~~~~~~~~~~~~~~~~~~~

**MXFP8 E4M3** provides ~2× memory reduction vs FP16:

.. code-block:: python

   import numpy as np
   from pychop import MXTensor
   
   # Large model weight (e.g., LLM)
   W = np.random.randn(4096, 4096).astype(np.float16)  # 32 MB
   
   # Quantize to MXFP8
   W_mx = MXTensor(W, format='mxfp8_e4m3')
   
   stats = W_mx.statistics()
   print(f"Original: {stats['fp16_memory_mb']:.2f} MB")
   print(f"MX: {stats['mx_memory_mb']:.2f} MB")
   print(f"Saved: {stats['memory_saved_vs_fp16']:.1f}%")

Accuracy vs Compression
------------------------

Trade-off between accuracy and compression:

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 45

   * - Format
     - Compress
     - Typical MSE
     - Best For
   * - MXFP8_E5M2
     - 1.94×
     - ~1e-3
     - Wide dynamic range (gradients)
   * - MXFP8_E4M3
     - 1.94×
     - ~1e-3
     - Balanced (weights, activations)
   * - MXFP6_E3M2
     - 2.56×
     - ~1e-2
     - Inference with high compression
   * - MXFP4_E2M1
     - 3.76×
     - ~1e-1
     - Extreme compression (careful!)

Training Recommendations
------------------------

For Quantization-Aware Training:

1. **Start with higher precision** (MXFP8) during early training
2. **Gradually reduce** to lower precision (MXFP6) if needed
3. **Use separate formats** for weights vs activations
4. **Monitor training loss** - if diverging, increase precision

.. code-block:: python

   from pychop.tch.mx_formats import MXLinear
   
   # Conservative: MXFP8 for weights and activations
   layer = MXLinear(768, 3072, weight_format='mxfp8_e4m3', act_format='mxfp8_e4m3')
   
   # Aggressive: MXFP6 for weights, MXFP8 for activations
   layer = MXLinear(768, 3072, weight_format='mxfp6_e3m2', act_format='mxfp8_e4m3')

Troubleshooting
---------------- 

Common Issues
~~~~~~~~~~~~~~~~~~~~~~

**Q: Getting NaN or Inf after quantization?**

A: Use higher precision format or smaller blocks:

.. code-block:: python

   # Instead of MXFP4
   X_q = mx_quantize(X, format='mxfp4_e2m1')  # May cause NaN
   
   # Use MXFP6 or MXFP8
   X_q = mx_quantize(X, format='mxfp6_e3m2')  # More stable

**Q: Training loss diverging with MX quantization?**

A: 

1. Use higher precision for gradients (don't quantize optimizer states)
2. Reduce learning rate
3. Use gradient clipping

.. code-block:: python

   # Only quantize forward pass
   x_q = mx_quantize(x, format='mxfp8_e4m3')
   y = model(x_q)
   loss = criterion(y, target)
   loss.backward()  # Gradients in full precision
   
   # Gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()

**Q: Overflow warnings in NumPy backend?**

A: This is normal for extreme values. The implementation automatically handles overflows, but you can suppress warnings:

.. code-block:: python

   import warnings
   with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       X_q = mx_quantize(X, format='mxfp8_e4m3')

**Q: Backend not detected automatically?**

A: Explicitly set backend:

.. code-block:: python

   import pychop
   pychop.backend('torch')  # Force PyTorch backend
   
   from pychop import mx_quantize
   X_q = mx_quantize(X, format='mxfp8_e4m3')

References
-----------------------------

Standards and Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **OCP Microscaling Formats (MX) v1.0 Specification**
   
   - Official spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
   - Published: 2023
   - Defines MXFP8, MXFP6, MXFP4, and MXINT8 formats

2. **IEEE 754 Floating-Point Standard**
   
   - Reference for floating-point arithmetic
   - https://ieeexplore.ieee.org/document/8766229

Research Papers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Microscaling Data Formats for Deep Learning** (Rouhani et al., 2023)
   
   - Introduces MX formats for neural networks
   - Shows minimal accuracy loss with 2-4× compression

2. **FP8 Formats for Deep Learning** (Micikevicius et al., 2022)
   
   - NVIDIA's work on 8-bit floating point
   - Demonstrates QAT effectiveness

3. **Mixed Precision Training** (Micikevicius et al., 2018)
   
   - Foundation for low-precision training
   - Loss scaling and gradient handling



Summary
--------

MX formats in Pychop provide:

 **OCP Standard Compliance** - Full support for all MX formats

 **Multi-Backend** - NumPy, PyTorch (STE), JAX (custom VJP)

 **Easy to Use** - Automatic backend detection, simple API

 **Production Ready** - QAT support, model conversion, LLM fine-tuning

 **Flexible** - Predefined + custom formats, configurable blocks

**Get Started:**

.. code-block:: python

   import pychop
   from pychop import mx_quantize
   
   pychop.backend('auto')  # Auto-detect
   
   # Quantize with MXFP8 E4M3
   X_q = mx_quantize(X, format='mxfp8_e4m3')
   
   # For PyTorch QAT
   from pychop.tch.mx_formats import MXLinear
   layer = MXLinear(768, 3072, weight_format='mxfp8_e4m3')

