.. _mx_formats:

Microscaling (MX) Formats
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


Supported Formats
-----------------

Predefined Formats
~~~~~~~~~~~~~~~~~~

The following standard OCP MX formats are available:

.. list-table:: Standard MX Formats
   :widths: 20 10 10 10 15 35
   :header-rows: 1

   * - Format Name
     - Bits
     - Exp
     - Sig
     - Block Size
     - Description
   * - ``mxfp8_e5m2``
     - 8
     - 5
     - 2
     - 32
     - NVIDIA FP8 format (E5M2 elements)
   * - ``mxfp8_e4m3``
     - 8
     - 4
     - 3
     - 32
     - NVIDIA FP8 format (E4M3 elements)
   * - ``mxfp6_e3m2``
     - 6
     - 3
     - 2
     - 32
     - 6-bit ultra-low precision
   * - ``mxfp6_e2m3``
     - 6
     - 2
     - 3
     - 32
     - 6-bit with more mantissa
   * - ``mxfp4_e2m1``
     - 4
     - 2
     - 1
     - 32
     - Extreme 4-bit format

Custom Formats
~~~~~~~~~~~~~~

**Pychop supports fully customizable MX formats!** You can specify:

- **Element format**: Any combination of exponent and significand bits
- **Scale format**: Customize the scale factor precision
- **Block size**: Adjust the number of elements per block

Three Ways to Create Custom Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Method 1: Tuple Format** (Simplest)

.. code-block:: python

    from pychop.mx_formats import MXTensor
    
    # E5M4 format (10-bit elements)
    mx = MXTensor(X, format=(5, 4))
    
    # E6M5 format (12-bit elements)
    mx = MXTensor(X, format=(6, 5))
    
    # E1M1 format (3-bit elements - extreme!)
    mx = MXTensor(X, format=(1, 1))

**Method 2: Using create_mx_spec** (More Control)

.. code-block:: python

    from pychop.mx_formats import create_mx_spec, MXTensor
    
    # Custom format with 10-bit scale
    spec = create_mx_spec(
        exp_bits=5,           # Element exponent bits
        sig_bits=4,           # Element significand bits
        scale_exp_bits=10,    # Scale exponent bits (default: 8)
        scale_sig_bits=0,     # Scale significand bits (default: 0)
        block_size=64,        # Block size (default: 32)
        name="MyCustomFormat" # Optional name
    )
    
    mx = MXTensor(X, format=spec)

**Method 3: Direct Parameters** (Most Flexible)

.. code-block:: python

    # Specify scale parameters directly in MXTensor
    mx = MXTensor(
        X,
        format=(4, 3),           # E4M3 elements
        block_size=32,
        scale_exp_bits=10,       # Use 10-bit exponent for scale
        scale_sig_bits=1         # Add 1-bit mantissa to scale
    )


API Reference
-------------

Core Classes
~~~~~~~~~~~~

MXSpec
^^^^^^

.. autoclass:: MXSpec
   :members:
   :undoc-members:

   **Attributes:**
   
   .. py:attribute:: name
      :type: str
      
      Format name (e.g., "MXFP8_E4M3")
   
   .. py:attribute:: exp_bits
      :type: int
      
      Number of exponent bits in each element
   
   .. py:attribute:: sig_bits
      :type: int
      
      Number of significand bits in each element (excluding sign bit)
   
   .. py:attribute:: scale_exp_bits
      :type: int
      
      Number of exponent bits in shared scale factor (default: 8)
   
   .. py:attribute:: scale_sig_bits
      :type: int
      
      Number of significand bits in shared scale factor (default: 0)
   
   .. py:attribute:: default_block_size
      :type: int
      
      Default number of elements per block (default: 32)
   
   .. py:attribute:: element_bits
      :type: int
      :annotation: [property]
      
      Total bits per element (1 sign + exp_bits + sig_bits)
   
   .. py:attribute:: scale_bits
      :type: int
      :annotation: [property]
      
      Total bits for scale factor (1 sign + scale_exp_bits + scale_sig_bits)

   **Example:**
   
   .. code-block:: python
   
       from pychop.mx_formats import MXSpec
       
       # Standard format
       spec = MXSpec("MXFP8_E4M3", exp_bits=4, sig_bits=3)
       
       # Custom format with all parameters
       spec = MXSpec(
           name="Custom_E5M4_S10",
           exp_bits=5,
           sig_bits=4,
           scale_exp_bits=10,
           scale_sig_bits=0,
           default_block_size=64
       )


MXBlock
^^^^^^^

.. autoclass:: MXBlock
   :members:
   :undoc-members:

   Represents a single microscaling block with shared scale factor.

   **Constructor:**

   .. code-block:: python

       MXBlock(values, spec, scale_factor=None, rmode=1, subnormal=True)

   **Parameters:**

   - **values** (*array-like*) – Input values to encode (typically 16-64 elements)
   - **spec** (*MXSpec*) – Format specification
   - **scale_factor** (*float, optional*) – Pre-computed scale factor (if None, computed automatically)
   - **rmode** (*int, default=1*) – Rounding mode for quantization
     
     - 1: Round to nearest, ties to even (IEEE 754 default)
     - 2: Round towards +∞
     - 3: Round towards -∞
     - 4: Truncate toward zero
     - 5: Stochastic rounding (proportional)
     - 6: Stochastic rounding (equal probability)
   
   - **subnormal** (*bool, default=True*) – Support subnormal numbers in elements

   **Attributes:**

   .. py:attribute:: scale_factor
      :type: float
      
      Shared scale factor for all elements in this block

   .. py:attribute:: elements
      :type: np.ndarray
      
      Quantized element values (low-precision)

   .. py:attribute:: original_values
      :type: np.ndarray
      
      Original input values before quantization

   **Methods:**

   .. py:method:: dequantize() -> np.ndarray
      
      Reconstruct original values from quantized representation.
      
      :returns: Dequantized values
      :rtype: np.ndarray

   .. py:method:: storage_bits() -> int
      
      Calculate total storage bits for this block.
      
      :returns: Total bits (scale_bits + len(elements) * element_bits)
      :rtype: int

   .. py:method:: compression_ratio(baseline_bits=16) -> float
      
      Calculate compression ratio compared to baseline format.
      
      :param baseline_bits: Baseline format bit-width (e.g., 16 for FP16, 32 for FP32)
      :type baseline_bits: int
      :returns: Compression ratio
      :rtype: float

   **Example:**

   .. code-block:: python

       from pychop.mx_formats import MXBlock, MX_FORMATS
       import numpy as np
       
       # Create a block
       values = np.array([1.5, 2.3, 0.8, -1.2, 3.7, 0.4])
       block = MXBlock(values, spec=MX_FORMATS['mxfp8_e4m3'])
       
       # Inspect
       print(f"Scale factor: {block.scale_factor}")
       print(f"Elements: {block.elements}")
       print(f"Storage: {block.storage_bits()} bits")
       print(f"Compression: {block.compression_ratio():.2f}x vs FP16")
       
       # Reconstruct
       reconstructed = block.dequantize()
       error = np.abs(values - reconstructed)
       print(f"Max error: {error.max():.6f}")


MXTensor
^^^^^^^^

.. autoclass:: MXTensor
   :members:
   :undoc-members:

   Multi-block microscaling tensor for encoding full arrays/tensors.

   **Constructor:**

   .. code-block:: python

       MXTensor(values, format='mxfp8_e4m3', block_size=None, axis=-1,
                rmode=1, subnormal=True, scale_exp_bits=None, scale_sig_bits=None)

   **Parameters:**

   - **values** (*array-like*) – Input tensor values (any shape)
   - **format** (*str, MXSpec, or tuple*) – Format specification:
     
     - String: ``'mxfp8_e4m3'`` (predefined format)
     - Tuple: ``(exp_bits, sig_bits)`` for quick custom format
     - MXSpec: custom specification object
   
   - **block_size** (*int, optional*) – Elements per block (default from spec, typically 32)
   - **axis** (*int, default=-1*) – Axis for blocking (-1 for flattened, other axes not yet implemented)
   - **rmode** (*int, default=1*) – Rounding mode (see MXBlock)
   - **subnormal** (*bool, default=True*) – Support subnormal numbers
   - **scale_exp_bits** (*int, optional*) – Override scale exponent bits (only for tuple format)
   - **scale_sig_bits** (*int, optional*) – Override scale significand bits (only for tuple format)

   **Attributes:**

   .. py:attribute:: spec
      :type: MXSpec
      
      Format specification being used

   .. py:attribute:: blocks
      :type: list[MXBlock]
      
      List of encoded blocks

   .. py:attribute:: shape
      :type: tuple
      
      Original tensor shape

   .. py:attribute:: block_size
      :type: int
      
      Number of elements per block

   **Methods:**

   .. py:method:: dequantize() -> np.ndarray
      
      Reconstruct full tensor from MX representation.
      
      :returns: Dequantized tensor with original shape
      :rtype: np.ndarray

   .. py:method:: storage_bits() -> int
      
      Total storage bits for entire tensor.
      
      :returns: Sum of storage bits across all blocks
      :rtype: int

   .. py:method:: compression_ratio(baseline_bits=16) -> float
      
      Calculate overall compression ratio.
      
      :param baseline_bits: Baseline format (16 for FP16, 32 for FP32)
      :type baseline_bits: int
      :returns: Compression ratio
      :rtype: float

   .. py:method:: statistics() -> dict
      
      Compute comprehensive statistics about the MX encoding.
      
      :returns: Dictionary with keys:
                
                - ``format``: Format name
                - ``element_bits``, ``exp_bits``, ``sig_bits``: Element format
                - ``scale_bits``: Scale factor bits
                - ``shape``: Tensor shape
                - ``n_blocks``: Number of blocks
                - ``block_size``: Block size
                - ``compression_ratio_fp16``, ``compression_ratio_fp32``: Compression ratios
                - ``storage_bits``, ``storage_bytes``: Storage requirements
                - ``mean_abs_error``, ``max_abs_error``: Absolute errors
                - ``mean_rel_error``, ``max_rel_error``: Relative errors
                - ``scale_min``, ``scale_max``, ``scale_mean``: Scale factor statistics
      
      :rtype: dict

   **Examples:**

   .. code-block:: python

       from pychop.mx_formats import MXTensor
       import numpy as np
       
       X = np.random.randn(128, 64)
       
       # Predefined format
       mx1 = MXTensor(X, format='mxfp8_e4m3', block_size=32)
       
       # Custom format (tuple)
       mx2 = MXTensor(X, format=(5, 4), block_size=64)
       
       # With custom scale
       mx3 = MXTensor(X, format=(4, 3), scale_exp_bits=10, block_size=32)
       
       # Reconstruct
       X_recon = mx1.dequantize()
       
       # Analyze
       stats = mx1.statistics()
       print(f"Compression: {stats['compression_ratio_fp16']:.2f}x")
       print(f"Mean error: {stats['mean_abs_error']:.6f}")
       print(f"Storage: {stats['storage_bytes']/1024:.2f} KB")


MXFloat
^^^^^^^

.. autoclass:: MXFloat
   :members:
   :undoc-members:

   Scalar value in MX format (single-element block). Similar to ``CPFloat`` but uses
   MX encoding.

   **Constructor:**

   .. code-block:: python

       MXFloat(value, format='mxfp8_e4m3')

   **Parameters:**

   - **value** (*float*) – Scalar value
   - **format** (*str, MXSpec, or tuple*) – MX format specification

   **Attributes:**

   .. py:attribute:: value
      :type: float
      :annotation: [property]
      
      Dequantized value

   .. py:attribute:: spec
      :type: MXSpec
      
      Format specification

   **Supported Operations:**

   - Arithmetic: ``+``, ``-``, ``*``, ``/`` (returns MXFloat)
   - Unary: ``-``, ``abs()``
   - Comparison: ``==``, ``<``, ``<=``, ``>``, ``>=``
   - Type conversion: ``float(mx_float)``

   **Examples:**

   .. code-block:: python

       from pychop.mx_formats import MXFloat
       
       # Create MXFloat numbers
       a = MXFloat(3.14159, 'mxfp8_e4m3')
       b = MXFloat(2.71828, 'mxfp8_e4m3')
       
       # Custom format
       c = MXFloat(1.41421, (5, 4))  # E5M4
       
       # Arithmetic (always returns MXFloat)
       d = a + b      # MXFloat(5.86, format=MXFP8_E4M3)
       e = a * b      # MXFloat(8.54, format=MXFP8_E4M3)
       f = a / 2.0    # Mixed with regular float
       
       # Comparisons
       print(a > b)   # True
       print(a == MXFloat(3.14, 'mxfp8_e4m3'))  # Close to True
       
       # Convert to float
       val = float(a)


Utility Functions
~~~~~~~~~~~~~~~~~

create_mx_spec
^^^^^^^^^^^^^^

.. autofunction:: create_mx_spec

   Create a custom MX format specification.

   **Signature:**

   .. code-block:: python

       create_mx_spec(exp_bits, sig_bits, scale_exp_bits=8, scale_sig_bits=0,
                      block_size=32, name=None) -> MXSpec

   **Parameters:**

   - **exp_bits** (*int*) – Exponent bits in each element
   - **sig_bits** (*int*) – Significand bits in each element
   - **scale_exp_bits** (*int, default=8*) – Exponent bits in scale factor
   - **scale_sig_bits** (*int, default=0*) – Significand bits in scale factor
   - **block_size** (*int, default=32*) – Default block size
   - **name** (*str, optional*) – Format name (auto-generated if None)

   **Returns:**

   Custom MX format specification

   **Return type:**

   MXSpec

   **Examples:**

   .. code-block:: python

       from pychop.mx_formats import create_mx_spec
       
       # Simple custom format
       spec1 = create_mx_spec(exp_bits=5, sig_bits=4)
       
       # With custom scale
       spec2 = create_mx_spec(
           exp_bits=4,
           sig_bits=3,
           scale_exp_bits=10,  # Larger dynamic range
           scale_sig_bits=1,   # More precision in scale
           block_size=64,
           name="MyFormat_E4M3_ScaleE10M1"
       )
       
       # Ultra-low precision
       spec3 = create_mx_spec(exp_bits=2, sig_bits=1, block_size=16)


mx_quantize
^^^^^^^^^^^

.. autofunction:: mx_quantize

   Convenience function to quantize array to MX format.

   **Signature:**

   .. code-block:: python

       mx_quantize(values, format='mxfp8_e4m3', block_size=None,
                   return_tensor=False, **kwargs) -> Union[np.ndarray, MXTensor]

   **Parameters:**

   - **values** (*array-like*) – Input values
   - **format** (*str, MXSpec, or tuple*) – MX format
   - **block_size** (*int, optional*) – Block size
   - **return_tensor** (*bool, default=False*) – If True, return MXTensor; else return dequantized array
   - **\\**kwargs** – Additional arguments passed to MXTensor (e.g., ``scale_exp_bits``)

   **Returns:**

   Quantized result (array or MXTensor object)

   **Return type:**

   np.ndarray or MXTensor

   **Examples:**

   .. code-block:: python

       from pychop.mx_formats import mx_quantize
       import numpy as np
       
       X = np.random.randn(1024)
       
       # Quick quantization (returns array)
       X_q = mx_quantize(X, 'mxfp8_e4m3', block_size=32)
       
       # Custom format
       X_q = mx_quantize(X, format=(5, 4), block_size=64)
       
       # Get MXTensor object
       mx_tensor = mx_quantize(X, format=(4, 3), return_tensor=True)
       print(mx_tensor.statistics())


compare_mx_formats
^^^^^^^^^^^^^^^^^^

.. autofunction:: compare_mx_formats

   Compare different MX formats on the same data.

   **Signature:**

   .. code-block:: python

       compare_mx_formats(values, formats=None, block_sizes=None,
                          custom_formats=None) -> dict

   **Parameters:**

   - **values** (*np.ndarray*) – Test data
   - **formats** (*list, optional*) – List of predefined format names
   - **block_sizes** (*list, optional*) – List of block sizes (default: [32])
   - **custom_formats** (*list of tuples, optional*) – List of ``(exp_bits, sig_bits)`` tuples

   **Returns:**

   Dictionary mapping format names to statistics dictionaries

   **Return type:**

   dict

   **Examples:**

   .. code-block:: python

       from pychop.mx_formats import compare_mx_formats
       import numpy as np
       
       X = np.random.randn(1024, 512)
       
       # Compare predefined formats
       results = compare_mx_formats(X, formats=['mxfp8_e4m3', 'mxfp6_e3m2'])
       
       # Compare custom formats
       results = compare_mx_formats(
           X,
           custom_formats=[(4, 3), (5, 4), (6, 5)]
       )
       
       # Compare block sizes
       results = compare_mx_formats(X, block_sizes=[16, 32, 64, 128])
       
       # Print results
       for fmt, stats in results.items():
           print(f"{fmt}: {stats['compression_ratio_fp16']:.2f}x, "
                 f"error={stats['mean_abs_error']:.6f}")


print_mx_format_table
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: print_mx_format_table

   Print a formatted table of available predefined MX formats.

   **Example:**

   .. code-block:: python

       from pychop.mx_formats import print_mx_format_table
       
       print_mx_format_table()

   **Output:**

   .. code-block:: text

       ====================================================================================================
       Available Microscaling (MX) Formats
       ====================================================================================================
       Format                    Element    Exp    Sig    Scale      Block    Compression (vs FP16)
       ----------------------------------------------------------------------------------------------------
       mxfp8_e5m2                8          5      2      E8M0       32       1.94x
       mxfp8_e4m3                8          4      3      E8M0       32       1.94x
       mxfp6_e3m2                6          3      2      E6M0       32       2.46x
       mxfp6_e2m3                6          2      3      E6M0       32       2.46x
       mxfp4_e2m1                4          2      1      E8M0       32       3.56x
       ====================================================================================================


Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

Quick Start
^^^^^^^^^^^

.. code-block:: python

    from pychop.mx_formats import MXTensor, mx_quantize
    import numpy as np
    
    # Generate data
    X = np.random.randn(1024, 512).astype(np.float32)
    
    # Method 1: Direct quantization
    X_q = mx_quantize(X, format='mxfp8_e4m3', block_size=32)
    error = np.mean(np.abs(X - X_q))
    print(f"Mean error: {error:.6f}")
    
    # Method 2: Keep MXTensor object for analysis
    mx_tensor = MXTensor(X, format='mxfp8_e4m3', block_size=32)
    print(f"Compression: {mx_tensor.compression_ratio():.2f}x")
    print(f"Storage: {mx_tensor.storage_bits() / 8 / 1024:.2f} KB")
    
    # Reconstruct
    X_recon = mx_tensor.dequantize()

Custom Formats
^^^^^^^^^^^^^^

**Example 1: Using Tuples**

.. code-block:: python

    # E5M4 format (10-bit elements)
    mx1 = MXTensor(X, format=(5, 4), block_size=32)
    
    # E6M5 format (12-bit elements)
    mx2 = MXTensor(X, format=(6, 5), block_size=64)
    
    # Ultra-low: E2M1 (4-bit elements)
    mx3 = MXTensor(X, format=(2, 1), block_size=16)

**Example 2: Custom Scale Factor**

.. code-block:: python

    # Standard E8M0 scale
    mx1 = MXTensor(X, format=(4, 3), block_size=32)
    
    # Larger dynamic range: E10M0 scale
    mx2 = MXTensor(X, format=(4, 3), block_size=32,
                   scale_exp_bits=10, scale_sig_bits=0)
    
    # Scale with mantissa: E8M2
    mx3 = MXTensor(X, format=(4, 3), block_size=32,
                   scale_exp_bits=8, scale_sig_bits=2)

**Example 3: Full Control with MXSpec**

.. code-block:: python

    from pychop.mx_formats import create_mx_spec
    
    spec = create_mx_spec(
        exp_bits=5,
        sig_bits=4,
        scale_exp_bits=10,
        scale_sig_bits=1,
        block_size=64,
        name="MyOptimalFormat"
    )
    
    mx = MXTensor(X, format=spec)
    stats = mx.statistics()
    print(f"Format: {stats['format']}")
    print(f"Compression: {stats['compression_ratio_fp16']:.2f}x")

Advanced Examples
~~~~~~~~~~~~~~~~~

Neural Network Layer Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from pychop.mx_formats import MXTensor
    
    # Simulate a neural network layer
    batch_size = 128
    input_dim = 512
    hidden_dim = 256
    
    # Layer parameters
    W = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b = np.random.randn(hidden_dim).astype(np.float32) * 0.01
    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    # FP32 forward pass
    Z_fp32 = X @ W + b
    A_fp32 = np.maximum(0, Z_fp32)  # ReLU
    
    # MX quantized forward pass
    W_mx_tensor = MXTensor(W, format='mxfp8_e4m3', block_size=32)
    X_mx_tensor = MXTensor(X, format='mxfp8_e4m3', block_size=32)
    b_mx_tensor = MXTensor(b, format='mxfp8_e4m3', block_size=32)
    
    W_mx = W_mx_tensor.dequantize()
    X_mx = X_mx_tensor.dequantize()
    b_mx = b_mx_tensor.dequantize()
    
    Z_mx = X_mx @ W_mx + b_mx
    A_mx = np.maximum(0, Z_mx)
    
    # Compare
    print(f"Weight compression: {W_mx_tensor.compression_ratio(32):.2f}x vs FP32")
    print(f"Weight storage: {W_mx_tensor.storage_bits() / 8 / 1024:.2f} KB")
    print(f"Activation error: {np.mean(np.abs(A_fp32 - A_mx)):.6f}")

Format Exploration
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pychop.mx_formats import compare_mx_formats
    
    # Test data with different characteristics
    X_normal = np.random.randn(2048)
    X_uniform = np.random.uniform(-10, 10, 2048)
    X_sparse = np.random.randn(2048)
    X_sparse[np.abs(X_sparse) < 1.0] = 0
    
    datasets = {
        'Normal': X_normal,
        'Uniform': X_uniform,
        'Sparse': X_sparse
    }
    
    # Compare different E/M combinations
    custom_formats = [
        (5, 2),  # More exponent (better range)
        (4, 3),  # Balanced
        (3, 4),  # More significand (better precision)
    ]
    
    for name, data in datasets.items():
        print(f"\n{name} distribution:")
        results = compare_mx_formats(data, custom_formats=custom_formats)
        
        for fmt, stats in results.items():
            print(f"  {fmt}: compression={stats['compression_ratio_fp16']:.2f}x, "
                  f"error={stats['mean_abs_error']:.6f}")

Block Size Optimization
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    X = np.random.randn(4096)
    
    block_sizes = [8, 16, 32, 64, 128]
    
    print(f"{'Block Size':<12} {'Compression':<15} {'Mean Error':<15} {'Max Error':<15}")
    print("-"*60)
    
    for bs in block_sizes:
        mx = MXTensor(X, format='mxfp8_e4m3', block_size=bs)
        stats = mx.statistics()
        
        print(f"{bs:<12} {stats['compression_ratio_fp16']:<15.2f} "
              f"{stats['mean_abs_error']:<15.6f} "
              f"{stats['max_abs_error']:<15.6f}")

MXFloat Scalar Arithmetic
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pychop.mx_formats import MXFloat
    
    # Create scalar values
    a = MXFloat(3.14159, 'mxfp8_e4m3')
    b = MXFloat(2.71828, 'mxfp8_e4m3')
    
    # Arithmetic operations
    c = a + b      # MXFloat(5.86, format=MXFP8_E4M3)
    d = a * b      # MXFloat(8.54, format=MXFP8_E4M3)
    e = a / 2.0    # Works with regular floats too
    
    # Complex expression
    result = (a + b) * c - d / 2.0
    print(f"Result: {result}")  # Still an MXFloat
    
    # Convert to regular float
    value = float(result)


Performance Considerations
--------------------------

Compression vs. Error Tradeoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 15 20 20 30
   :header-rows: 1

   * - Format
     - Element Bits
     - Compression (vs FP16)
     - Typical Error
     - Best For
   * - E5M2 (8-bit)
     - 8
     - ~1.9x
     - ~10⁻³
     - High dynamic range
   * - E4M3 (8-bit)
     - 8
     - ~1.9x
     - ~10⁻⁴
     - Balanced (most common)
   * - E3M2 (6-bit)
     - 6
     - ~2.5x
     - ~10⁻³
     - Moderate compression
   * - E2M1 (4-bit)
     - 4
     - ~3.6x
     - ~10⁻²
     - Extreme compression
   * - E5M4 (10-bit)
     - 10
     - ~1.6x
     - ~10⁻⁵
     - High precision

Block Size Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**Larger blocks** (64-128 elements):

- ✅ Better compression ratio (amortize scale overhead)
- ⚠️ May reduce precision if data has varying magnitudes
- 🎯 Good for: static weights, uniform data

**Smaller blocks** (16-32 elements):

- ✅ Better precision for diverse data
- ⚠️ More scale overhead
- 🎯 Good for: activations, gradients, dynamic data

**Recommendation**: Start with **block_size=32** (standard), then tune based on your data.

Scale Factor Selection
~~~~~~~~~~~~~~~~~~~~~~

**Standard E8M0** (8-bit exponent, 0 mantissa):

- ✅ Works for most cases
- Dynamic range: 2⁻¹²⁷ to 2¹²⁷
- Overhead: 8 bits per block

**Large Range E10M0** (10-bit exponent):

- ✅ Better for extreme dynamic ranges
- Dynamic range: 2⁻⁵¹¹ to 2⁵¹¹
- Overhead: 10 bits per block
- 🎯 Use when: data spans many orders of magnitude

**With Mantissa E8M2** (8-bit exp, 2-bit mantissa):

- ✅ More precise scale factors
- ⚠️ More overhead (10 bits)
- 🎯 Use when: precision is critical

Design Guidelines
~~~~~~~~~~~~~~~~~

1. **Choose Element Format**

   - More **exponent bits** → better for large dynamic range
   - More **significand bits** → better for precision
   - Start with **E4M3** (balanced)

2. **Choose Block Size**

   - Uniform data → larger blocks (64-128)
   - Diverse data → smaller blocks (16-32)
   - Default: **32 elements**

3. **Choose Scale Format**

   - Standard data → **E8M0** (default)
   - Extreme ranges → **E10M0**
   - High precision → **E8M2**

4. **Test and Iterate**

   - Use ``compare_mx_formats()`` to test multiple configurations
   - Check ``statistics()`` for error analysis
   - Balance compression vs. accuracy for your application


Application Examples
--------------------

Weight Quantization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Neural network weights (static, need precision)
    W = np.random.randn(1024, 512) * 0.1
    
    # Test formats
    formats_to_test = [
        ('mxfp8_e4m3', "Standard 8-bit"),
        ((5, 4), "Custom E5M4 (10-bit)"),
        ((6, 5), "High precision E6M5 (12-bit)")
    ]
    
    print(f"{'Format':<30} {'Storage (KB)':<15} {'Compression':<15} {'Error':<15}")
    print("-"*75)
    
    for fmt, desc in formats_to_test:
        mx = MXTensor(W, format=fmt, block_size=64)  # Larger blocks for weights
        stats = mx.statistics()
        
        print(f"{desc:<30} {stats['storage_bytes']/1024:<15.2f} "
              f"{stats['compression_ratio_fp32']:<15.2f} "
              f"{stats['mean_abs_error']:<15.6f}")

Activation Quantization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Activations (dynamic, varying magnitudes)
    A = np.maximum(0, np.random.randn(128, 512) * 10)  # ReLU output
    
    # Prefer formats with good dynamic range
    mx = MXTensor(A, format='mxfp8_e5m2', block_size=32)  # E5M2 for range
    
    stats = mx.statistics()
    print(f"Compression: {stats['compression_ratio_fp16']:.2f}x")
    print(f"Error: {stats['mean_abs_error']:.6f}")

Gradient Quantization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Gradients (sparse, extreme dynamic range)
    G = np.random.randn(1024, 512) * 0.001
    G[np.random.rand(1024, 512) > 0.9] *= 1000  # Some large gradients
    
    # Need large dynamic range
    mx = MXTensor(G, format=(5, 3), block_size=32,  # E5M3
                  scale_exp_bits=10)  # Larger scale range
    
    stats = mx.statistics()
    print(f"Format: {stats['format']}")
    print(f"Scale range: [{stats['scale_min']:.2e}, {stats['scale_max']:.2e}]")
    print(f"Relative error: {stats['mean_rel_error']:.4%}")


References
----------

Standards and Papers
~~~~~~~~~~~~~~~~~~~~

1. **OCP Microscaling Formats Specification v1.0**
   
   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

2. **Microscaling Data Formats for Deep Learning** (Microsoft Research, 2023)
   
   Rouhani et al., arXiv:2310.10537
   
   https://arxiv.org/abs/2310.10537

3. **OCP 8-bit Floating Point Specification (OFP8)**
   
   https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf

4. **IEEE 754 Floating-Point Standard**
   
   IEEE Std 754-2019

Related Libraries
~~~~~~~~~~~~~~~~~

- **QPyTorch**: PyTorch-specific low-precision simulation
- **gfloat**: Generic floating-point type library (research-oriented)
- **ml_dtypes**: JAX/NumPy low-precision types (performance-oriented)
- **Microsoft MX PyTorch Emulation**: Official MX format implementation


See Also
--------

- :ref:`builtin` - CPFloat and other built-in types
- :ref:`chop` - Core quantization functions
- :ref:`layers` - Quantized neural network layers
- :ref:`examples` - More usage examples


.. note::
   
   **New Feature!** Full customization of E/M combinations is supported since version 0.4.5.
   You can now create any MX format you need for your specific application!

.. warning::
   
   Extreme low-precision formats (< 4 bits per element) may result in significant accuracy
   loss. Always validate with your specific data before deploying to production.

.. tip::
   
   Use ``mx.statistics()`` to get comprehensive analysis of your MX encoding, including
   compression ratio, error metrics, and scale factor distribution.
