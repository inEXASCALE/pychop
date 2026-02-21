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

.. _fixed_point_simulator:



The `Chopf` class enables the quantization of floating-point numbers into a fixed-point Qm.n format, where `m` is the number of integer bits (including the sign bit) and `n` is the number of fractional bits. This document describes the usage and provides examples for implementations in PyTorch, NumPy, and JAX, each supporting six rounding modes: `nearest`, `up`, `down`, `towards_zero`, `stochastic_equal`, and `stochastic_proportional`.

Overview
--------

The simulator converts floating-point values into a fixed-point representation with a resolution of \(2^{-n}\) and a range of \([-2^{m-1}, 2^{m-1} - 2^{-n}]\). For the Q4.4 format used in the examples:
- **Resolution**: \(2^{-4} = 0.0625\)
- **Range**: \([-8.0, 7.9375]\)

The quantization process scales the input by the resolution, applies the chosen rounding mode, reconstructs the fixed-point value, and clamps it to the valid range.

Usage
-----

Common Parameters
~~~~~~~~~~~~~~~~~

- **ibits**: Specifies the number of bits for the integer part, including the sign bit.
- **fbits**: Defines the number of bits for the fractional part.
- **rmode**: Selects the rounding method, defaulting to "nearest".

PyTorch Version
~~~~~~~~~~~~~~~

The PyTorch implementation integrates with PyTorch tensors, making it suitable for machine learning workflows.

**Initialization**

Create an instance by setting the integer and fractional bit counts to define the Qm.n format.

**Quantization**

Quantize a tensor of floating-point values by invoking the quantization method, optionally specifying a rounding mode. The result is a tensor with quantized values.

**Code Example**:

.. code-block:: python

    # Initialize with Q4.4 format
    sim = Chopf(ibits=4, fbits=4)
    # Input tensor
    values = torch.tensor([1.7641, 0.3097, -0.2021, 2.4700, 0.3300])
    # Quantize with nearest rounding
    result = sim.quantize(values, rounding_mode="nearest")
    print(result)

NumPy Version
~~~~~~~~~~~~~

The NumPy version operates on NumPy arrays, offering a general-purpose quantization tool.

**Initialization**

Instantiate the simulator with the desired integer and fractional bit counts.

**Quantization**

Apply the quantization method to a NumPy array, with an optional rounding mode parameter, to obtain a quantized array.

**Code Example**:

.. code-block:: python

    # Initialize with Q4.4 format
    sim = Chopf(ibits=4, fbits=4)
    # Input array
    values = np.array([1.7641, 0.3097, -0.2021, 2.4700, 0.3300])
    # Quantize with nearest rounding
    result = sim.quantize(values, rounding_mode="nearest")
    print(result)

JAX Version
~~~~~~~~~~~

The JAX implementation uses JAX arrays and includes JIT compilation for performance, requiring a PRNG key for stochastic modes.

**Initialization**

Set up the simulator by defining the integer and fractional bits for the Qm.n format.

**Quantization**

Quantize a JAX array using the quantization method, specifying a rounding mode and, for stochastic modes, a PRNG key. The output is a quantized JAX array.

**Code Example**:

.. code-block:: python

    # Initialize with Q4.4 format
    sim = FPRound(ibits=4, fbits=4)
    # Input array
    values = jnp.array([1.7641, 0.3097, -0.2021, 2.4700, 0.3300])
    # PRNG key for stochastic modes
    key = random.PRNGKey(42)
    # Quantize with nearest rounding (no key needed)
    result = sim.quantize(values, rounding_mode="nearest")
    print(result)

Examples
--------

The following examples show the quantization of the input values `[1.7641, 0.3097, -0.2021, 2.47, 0.33]` in Q4.4 format across all rounding modes, consistent across PyTorch, NumPy, and JAX (with JAX using PRNG key 42 for stochastic modes).

**Input Values**

.. code-block:: text

    [1.7641, 0.3097, -0.2021, 2.47, 0.33]

**Outputs by Rounding Mode**

- **Nearest**:

  .. code-block:: text

      [1.75, 0.3125, -0.1875, 2.5, 0.3125]

  Rounds to the nearest representable value.

- **Up**:

  .. code-block:: text

      [1.8125, 0.3125, -0.1875, 2.5, 0.375]

  Positive values round toward positive infinity, negative values toward negative infinity.

- **Down**:

  .. code-block:: text

      [1.75, 0.25, -0.25, 2.4375, 0.3125]

  Positive values round toward negative infinity, negative values toward positive infinity.

- **Towards Zero**:

  .. code-block:: text

      [1.75, 0.25, -0.1875, 2.4375, 0.3125]

  Truncates toward zero, reducing the magnitude.

- **Stochastic Equal**:

  .. code-block:: text

      [1.8125, 0.3125, -0.25, 2.5, 0.3125]

  Randomly selects between floor and ceiling with equal probability (example with JAX PRNG key 42; varies otherwise).

- **Stochastic Proportional**:

  .. code-block:: text

      [1.8125, 0.3125, -0.1875, 2.4375, 0.3125]

  Randomly selects between floor and ceiling, with probability proportional to the fractional part (example with JAX PRNG key 42; varies otherwise).



This guide provides a clear introduction to using the `FPRound` classes, with practical examples formatted as code blocks for clarity.
