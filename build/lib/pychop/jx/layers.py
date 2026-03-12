"""JAX/Flax backend implementation for quantized neural network layers.

This module provides Flax-compatible quantized layers for quantization-aware
training (QAT) with JAX. All layers support custom quantizers via the `chop`
parameter and implement straight-through estimators (STE) for gradient flow.

Classes
-------
ChopSTE : Chop with STE for floating-point quantization
ChopfSTE : Fixed-point quantization with STE
ChopiSTE : Integer quantization with STE
QuantizedDense : Quantized fully-connected layer
QuantizedConv1d, QuantizedConv2d, QuantizedConv3d : Quantized convolution layers
QuantizedReLU, QuantizedGELU, QuantizedSoftmax : Quantized activation layers
QuantizedBatchNorm, QuantizedLayerNorm : Quantized normalization layers
QuantizedDropout : Quantized dropout layer

Functions
---------
post_quantization : Post-training quantization for Flax models

Notes
-----
- All layer APIs mirror PyTorch conventions for consistency
- Not all PyTorch layers are implemented yet (RNN, LSTM, Attention, etc.)
- Unimplemented layers raise NotImplementedError with guidance
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Any, Callable, Tuple, Union
import copy

# Import quantizers from parent module
from ..chop import Chop
from ..integer import Chopi
from ..fixed_point import Chopf



# ===================================================================
# Straight-Through Estimator (STE) Implementation for JAX
# ===================================================================

def create_ste_quantizer(quantize_fn):
    """Create a quantization function with straight-through estimator.
    
    This factory function creates a custom VJP quantization function that
    uses the provided quantize_fn in the forward pass but passes gradients
    straight through in the backward pass.
    
    Parameters
    ----------
    quantize_fn : callable
        Quantization function that takes an array and returns quantized array.
    
    Returns
    -------
    callable
        A function with STE that can be used in JAX transformations.
    
    Notes
    -----
    This uses closure to capture the quantize_fn, avoiding the need to pass
    non-JAX-type objects through custom_vjp.
    """
    
    @jax.custom_vjp
    def quantize_with_ste(x):
        """Apply quantization with STE."""
        return quantize_fn(x)
    
    def quantize_fwd(x):
        """Forward pass: apply quantization."""
        return quantize_fn(x), (x,)
    
    def quantize_bwd(res, g):
        """Backward pass: pass gradient straight through."""
        x, = res
        return (g,)
    
    quantize_with_ste.defvjp(quantize_fwd, quantize_bwd)
    
    return quantize_with_ste


# ===================================================================
# Quantization Wrappers with STE
# ===================================================================

class ChopSTE(Chop):
    """Floating-point quantizer with straight-through estimator for JAX QAT.

    This class wraps the base Chop quantizer with automatic STE application
    during training, enabling quantization-aware training with JAX/Flax.

    Parameters
    ----------
    exp_bits : int
        Number of exponent bits in the target floating-point format.
    sig_bits : int
        Number of significand bits (including implicit leading bit).
    rmode : int, default=1
        Rounding mode:
        - 1: Round to nearest, ties to even (IEEE 754 default)
        - 2: Round toward +infinity
        - 3: Round toward -infinity
        - 4: Round toward zero
        - 5: Stochastic rounding (proportional)
        - 6: Stochastic rounding (50/50)
        - 7: Round to nearest, ties to zero
        - 8: Round to nearest, ties away from zero
        - 9: Round to odd
    subnormal : bool, default=True
        Whether to support subnormal numbers.
    chunk_size : int, default=800
        Chunk size for processing large arrays.
    random_state : int, default=42
        Random seed for stochastic rounding modes.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    u : float
        Unit roundoff of the simulated floating-point format.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from pychop.jx.layers import ChopSTE
    >>> chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> quantized = chop(x)
    >>> # Gradients flow through during backprop

    Notes
    -----
    Unlike PyTorch, JAX doesn't have a `requires_grad` attribute, so STE
    is always applied. This is appropriate since JAX transformations (jit,
    grad, etc.) handle automatic differentiation transparently.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create STE version of quantization using closure
        def _quantize(x):
            return Chop.__call__(self, x)
        
        self._quantize_with_ste = create_ste_quantizer(_quantize)

    def __call__(self, x):
        """Apply quantization with STE.

        Parameters
        ----------
        x : jax.Array
            Input array to quantize.

        Returns
        -------
        jax.Array
            Quantized output with STE for gradient flow.
        """
        return self._quantize_with_ste(x)


class ChopfSTE(Chopf):
    """Fixed-point quantizer with straight-through estimator for JAX QAT.

    This class wraps the base Chopf fixed-point quantizer with automatic STE
    application, enabling fixed-point quantization-aware training with JAX/Flax.

    Parameters
    ----------
    ibits : int, default=4
        Number of bits for integer part (including sign bit if signed).
    fbits : int, default=4
        Number of bits for fractional part.
    rmode : int, default=1
        Rounding mode (same options as ChopSTE).

    Examples
    --------
    >>> from pychop.jx.layers import ChopfSTE
    >>> chop = ChopfSTE(ibits=8, fbits=8)  # Q8.8 fixed-point
    >>> x = jnp.array([1.5, 2.25, 3.75])
    >>> quantized = chop(x)

    Notes
    -----
    Fixed-point representation: value = integer_part.fractional_part
    Total bits = ibits + fbits (sign bit is included in ibits if signed).
    """

    def __init__(self, ibits: int = 4, fbits: int = 4, rmode: int = 1):
        super().__init__(ibits=ibits, fbits=fbits, rmode=rmode)
        
        # Create STE version of quantization using closure
        def _quantize(x):
            return Chopf.quantize(self, x)
        
        self._quantize_with_ste = create_ste_quantizer(_quantize)

    def __call__(self, X):
        """Apply fixed-point quantization with STE.

        Parameters
        ----------
        X : jax.Array or array-like
            Input array to quantize.

        Returns
        -------
        jax.Array
            Quantized output with STE for gradient flow.
        """
        if hasattr(X, 'ndim'):  # Check if it's a JAX array
            return self._quantize_with_ste(X)
        return Chopf.quantize(self, X)

    def quantize(self, X):
        """Alias for __call__ (for backward compatibility).

        Parameters
        ----------
        X : jax.Array or array-like
            Input array to quantize.

        Returns
        -------
        jax.Array
            Quantized output.
        """
        return self(X)


class ChopiSTE(Chopi):
    """Integer quantizer with straight-through estimator for JAX QAT.

    This class wraps the base Chopi integer quantizer with automatic STE
    application, performing fake quantization (quantize-dequantize) for
    integer quantization-aware training.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for integer representation.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    **kwargs
        Additional arguments passed to base Chopi class.

    Examples
    --------
    >>> from pychop.jx.layers import ChopiSTE
    >>> chop = ChopiSTE(bits=8, symmetric=True)
    >>> x = jnp.array([1.5, 2.25, 3.75])
    >>> fake_quantized = chop(x)  # Quantized to 8-bit then converted back

    Notes
    -----
    Performs fake quantization: x -> quantize(x) -> dequantize(quantize(x))
    This maintains floating-point representation while simulating integer
    quantization effects during training.
    """

    def __init__(self, bits: int = 8, symmetric: bool = False, **kwargs):
        super().__init__(bits=bits, symmetric=symmetric, **kwargs)
        
        # Create STE version of fake quantization using closure
        def _fake_quantize(x):
            q = self.quantize(x)
            dq = self.dequantize(q)
            return dq
        
        self._quantize_with_ste = create_ste_quantizer(_fake_quantize)

    def __call__(self, x):
        """Apply fake integer quantization with STE.

        Parameters
        ----------
        x : jax.Array
            Input array to fake-quantize.

        Returns
        -------
        jax.Array
            Fake-quantized output (quantize-dequantize cycle).

        Notes
        -----
        Returns input unchanged if x is None or empty.
        """
        if x is None or (hasattr(x, 'size') and x.size == 0):
            return x

        return self._quantize_with_ste(x)


# ===================================================================
# Post-Training Quantization for JAX/Flax
# ===================================================================

def post_quantization(model, chop, eval_mode: bool = True, verbose: bool = False):
    """Perform post-training quantization on a Flax model or parameter pytree.

    This function quantizes all weights and biases in a model while preserving
    other parameters (e.g., batch normalization statistics) unchanged.

    Parameters
    ----------
    model : flax.linen.Module or dict
        Either a Flax model instance with a `params` attribute, or a raw
        parameter pytree (nested dict/tuple structure).
    chop : Chop, Chopf, or Chopi
        Quantizer instance with a `quantize()` method or callable interface.
    eval_mode : bool, default=True
        Included for API consistency with PyTorch backend. Has no effect
        in JAX since evaluation vs. training is controlled per-call.
    verbose : bool, default=False
        If True, print names and shapes of quantized parameters.

    Returns
    -------
    quantized_model : flax.linen.Module or dict
        If input was a Flax model, returns model with quantized parameters.
        If input was a pytree, returns quantized pytree.

    Examples
    --------
    >>> import flax.linen as nn
    >>> from pychop.jx.layers import ChopSTE, post_quantization
    >>> 
    >>> # Define a simple model
    >>> class CNN(nn.Module):
    ...     @nn.compact
    ...     def __call__(self, x):
    ...         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    ...         x = nn.relu(x)
    ...         return x
    >>> 
    >>> # Initialize model and quantize
    >>> model = CNN()
    >>> params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
    >>> chop = Chop(exp_bits=5, sig_bits=10)  # Use base Chop, not ChopSTE
    >>> quantized_params = post_quantization(params, chop, verbose=True)

    Notes
    -----
    - Only parameters with 'kernel', 'weight', or 'bias' in their path are quantized
    - Other parameters (e.g., 'scale', 'mean', 'var' for batch norm) are preserved
    - Shape consistency is automatically verified
    - Compatible with any JAX pytree structure
    - If input contains both 'params' and 'batch_stats', both are returned
    """
    import jax.tree_util as tree_util

    def quantize_leaf(path, param):
        """Quantize a single parameter if it's a weight or bias.

        Parameters
        ----------
        path : list of str
            Path to the parameter in the pytree.
        param : jax.Array
            Parameter value to potentially quantize.

        Returns
        -------
        jax.Array
            Quantized parameter if it's a weight/bias, otherwise unchanged.
        """
        # Convert path to string list
        path_str = [str(k.key) if hasattr(k, 'key') else str(k) for k in path]
        path_str_joined = '/'.join(path_str)
        
        # Only quantize weights and biases (not batch norm statistics)
        # Skip batch_stats entirely
        if 'batch_stats' in path_str_joined:
            return param
        
        if any(name in path_str for name in ['kernel', 'weight', 'bias']):
            # Use the quantize method if available, otherwise call directly
            # For PTQ, we want to use the base quantization without STE
            if isinstance(chop, ChopSTE):
                # For ChopSTE in PTQ, call parent class directly
                quantized = Chop.__call__(chop, param)
            elif isinstance(chop, ChopfSTE):
                # For ChopfSTE in PTQ, call parent class directly
                quantized = Chopf.quantize(chop, param)
            elif isinstance(chop, ChopiSTE):
                # For ChopiSTE in PTQ, use fake quantization
                q = chop.quantize(param)
                quantized = chop.dequantize(q)
            elif hasattr(chop, 'quantize'):
                quantized = chop.quantize(param)
            else:
                quantized = chop(param)
            
            if verbose:
                print(f"[Quantized] {path_str_joined}: shape {param.shape}")
            return quantized
        return param

    # Check if input is a dict with 'params' and 'batch_stats'
    if isinstance(model, dict):
        if 'params' in model:
            # It's a variables dict with params and possibly batch_stats
            quantized_params = tree_util.tree_map_with_path(
                quantize_leaf,
                model['params']
            )
            
            result = {'params': quantized_params}
            
            # Preserve batch_stats unchanged if present
            if 'batch_stats' in model:
                result['batch_stats'] = model['batch_stats']
                if verbose:
                    print("\n[Info] batch_stats preserved (not quantized)")
            
            return result
        else:
            # It's a raw params pytree
            quantized_params = tree_util.tree_map_with_path(
                quantize_leaf,
                model
            )
            return quantized_params
    elif hasattr(model, 'params'):
        # It's a Flax model with a params attribute
        quantized_params = tree_util.tree_map_with_path(
            quantize_leaf,
            model.params
        )
        model = model.replace(params=quantized_params)
        return model
    else:
        # Treat as raw pytree
        quantized_params = tree_util.tree_map_with_path(
            quantize_leaf,
            model
        )
        return quantized_params
  
# ===================================================================
# Quantized Layers (Flax Implementation)
# ===================================================================

class QuantizedDense(nn.Module):
    """Quantized fully-connected (Dense) layer for JAX/Flax QAT.

    This layer applies quantization to weights, biases, and activations
    during forward pass, enabling quantization-aware training.

    Parameters
    ----------
    features : int
        Number of output features.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance. If None, layer behaves as standard Dense.
    dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the computation.
    param_dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the parameters.
    kernel_init : callable, default=lecun_normal()
        Initializer for the kernel (weight matrix).
    bias_init : callable, default=zeros
        Initializer for the bias vector.

    Attributes
    ----------
    features : int
        Number of output features.

    Examples
    --------
    >>> from pychop.jx.layers import QuantizedDense, ChopSTE
    >>> import jax.numpy as jnp
    >>> import jax
    >>> 
    >>> chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> layer = QuantizedDense(features=128, chop=chop)
    >>> 
    >>> # Initialize
    >>> x = jnp.ones((32, 64))  # batch_size=32, input_dim=64
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> 
    >>> # Forward pass
    >>> output = layer.apply(variables, x)
    >>> output.shape  # (32, 128)

    Notes
    -----
    Quantization order:
    1. Quantize kernel (weights)
    2. Quantize bias (if use_bias=True)
    3. Compute y = x @ kernel + bias
    4. Quantize output y
    """

    features: int
    use_bias: bool = True
    chop: Optional[Any] = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized dense layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch_size, input_dim).

        Returns
        -------
        jax.Array
            Output array of shape (batch_size, features).
        """
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (inputs.shape[-1], self.features),
            self.param_dtype
        )

        if self.chop is not None:
            kernel = self.chop(kernel)

        y = jnp.dot(inputs, kernel)

        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
            if self.chop is not None:
                bias = self.chop(bias)
            y = y + bias

        if self.chop is not None:
            y = self.chop(y)

        return jnp.asarray(y, self.dtype)


# Alias for consistency with PyTorch naming
QuantizedLinear = QuantizedDense


class QuantizedConv(nn.Module):
    """Base quantized convolution layer for JAX/Flax QAT.

    This is a generic n-dimensional quantized convolution layer that can
    handle 1D, 2D, and 3D convolutions.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolution kernel (e.g., (3, 3) for 2D).
    strides : tuple of int, optional
        Stride of the convolution. If None, defaults to (1, ..., 1).
    padding : str or tuple, default='SAME'
        Padding mode. Either 'SAME', 'VALID', or explicit padding tuple.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.
    dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the computation.
    param_dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the parameters.
    kernel_init : callable, default=lecun_normal()
        Initializer for the kernel.
    bias_init : callable, default=zeros
        Initializer for the bias.

    Notes
    -----
    This is a base class. Use QuantizedConv1d, QuantizedConv2d, or
    QuantizedConv3d for specific dimensions.
    """

    features: int
    kernel_size: tuple
    strides: Optional[tuple] = None
    padding: Union[str, tuple] = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized convolution layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, spatial_dims..., channels).

        Returns
        -------
        jax.Array
            Output array of shape (batch, spatial_dims..., features).
        """
        if self.strides is None:
            strides = (1,) * (inputs.ndim - 2)
        else:
            strides = self.strides

        kernel_shape = self.kernel_size + (inputs.shape[-1], self.features)
        kernel = self.param(
            'kernel',
            self.kernel_init,
            kernel_shape,
            self.param_dtype
        )

        if self.chop is not None:
            kernel = self.chop(kernel)

        # Determine dimension numbers based on input shape
        # Flax convention: (batch, spatial..., channels)
        spatial_dims = inputs.ndim - 2
        if spatial_dims == 1:
            dimension_numbers = ('NHC', 'HIO', 'NHC')
        elif spatial_dims == 2:
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        elif spatial_dims == 3:
            dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

        y = jax.lax.conv_general_dilated(
            inputs,
            kernel,
            window_strides=strides,
            padding=self.padding,
            dimension_numbers=dimension_numbers
        )

        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
            if self.chop is not None:
                bias = self.chop(bias)
            y = y + bias

        if self.chop is not None:
            y = self.chop(y)

        return jnp.asarray(y, self.dtype)


class QuantizedConv1d(nn.Module):
    """Quantized 1D convolution layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int
        Size of the convolution kernel.
    strides : int, default=1
        Stride of the convolution.
    padding : str or int, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> from pychop.jx.layers import QuantizedConv1d, ChopSTE
    >>> chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> layer = QuantizedConv1d(features=64, kernel_size=3, chop=chop)
    >>> x = jnp.ones((32, 100, 16))  # (batch, length, channels)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    >>> output.shape  # (32, 100, 64) with 'SAME' padding
    """

    features: int
    kernel_size: int
    strides: int = 1
    padding: Union[str, int] = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized 1D convolution layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, length, in_channels).

        Returns
        -------
        jax.Array
            Output array of shape (batch, length, features).
        """
        return QuantizedConv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding=self.padding,
            use_bias=self.use_bias,
            chop=self.chop
        )(inputs)


class QuantizedConv2d(nn.Module):
    """Quantized 2D convolution layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel. If int, assumes square kernel.
    strides : int or tuple of int, default=(1, 1)
        Stride of the convolution.
    padding : str or tuple, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> from pychop.jx.layers import QuantizedConv2d, ChopSTE
    >>> chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> layer = QuantizedConv2d(features=32, kernel_size=3, chop=chop)
    >>> x = jnp.ones((32, 28, 28, 1))  # (batch, height, width, channels)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    >>> output.shape  # (32, 28, 28, 32) with 'SAME' padding
    """

    features: int
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: Union[str, Tuple] = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized 2D convolution layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, height, width, in_channels).

        Returns
        -------
        jax.Array
            Output array of shape (batch, height, width, features).
        """
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if isinstance(self.strides, int):
            strides = (self.strides, self.strides)
        else:
            strides = self.strides

        return QuantizedConv(
            features=self.features,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=self.chop
        )(inputs)


class QuantizedConv3d(nn.Module):
    """Quantized 3D convolution layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel. If int, assumes cubic kernel.
    strides : int or tuple of int, default=(1, 1, 1)
        Stride of the convolution.
    padding : str or tuple, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> from pychop.jx.layers import QuantizedConv3d, ChopSTE
    >>> chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> layer = QuantizedConv3d(features=16, kernel_size=3, chop=chop)
    >>> x = jnp.ones((8, 16, 16, 16, 1))  # (batch, depth, height, width, channels)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    >>> output.shape  # (8, 16, 16, 16, 16)
    """

    features: int
    kernel_size: Union[int, Tuple[int, int, int]]
    strides: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    padding: Union[str, Tuple] = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized 3D convolution layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, depth, height, width, in_channels).

        Returns
        -------
        jax.Array
            Output array of shape (batch, depth, height, width, features).
        """
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if isinstance(self.strides, int):
            strides = (self.strides, self.strides, self.strides)
        else:
            strides = self.strides

        return QuantizedConv(
            features=self.features,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=self.chop
        )(inputs)


# ===================================================================
# Quantized Activation Layers
# ===================================================================

class QuantizedReLU(nn.Module):
    """Quantized ReLU activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance. If None, layer behaves as standard ReLU.

    Examples
    --------
    >>> from pychop.jx.layers import QuantizedReLU, ChopSTE
    >>> chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> layer = QuantizedReLU(chop=chop)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> # In Flax, activation layers are typically stateless
    >>> output = layer(x)  # No need for init/apply
    """

    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized ReLU layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with ReLU applied and quantized.
        """
        y = nn.relu(inputs)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedSigmoid(nn.Module):
    """Quantized Sigmoid activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedSigmoid(chop=chop)
    >>> output = layer(x)
    """

    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized sigmoid layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with sigmoid applied and quantized.
        """
        y = nn.sigmoid(inputs)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedTanh(nn.Module):
    """Quantized Tanh activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedTanh(chop=chop)
    >>> output = layer(x)
    """

    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized tanh layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with tanh applied and quantized.
        """
        y = nn.tanh(inputs)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedGELU(nn.Module):
    """Quantized GELU activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.
    approximate : bool, default=False
        Whether to use approximate GELU implementation.

    Examples
    --------
    >>> layer = QuantizedGELU(chop=chop, approximate=False)
    >>> output = layer(x)

    Notes
    -----
    GELU (Gaussian Error Linear Unit) is commonly used in transformers
    and other modern architectures.
    """

    chop: Optional[Any] = None
    approximate: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized GELU layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with GELU applied and quantized.
        """
        y = nn.gelu(inputs, approximate=self.approximate)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedSoftmax(nn.Module):
    """Quantized Softmax activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.
    axis : int, default=-1
        Axis along which to apply softmax.

    Examples
    --------
    >>> layer = QuantizedSoftmax(chop=chop, axis=-1)
    >>> logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> probs = layer(logits)

    Notes
    -----
    Softmax is typically used as the final layer in classification tasks.
    Quantizing softmax output may affect numerical stability.
    """

    chop: Optional[Any] = None
    axis: int = -1

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized softmax layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array (logits).

        Returns
        -------
        jax.Array
            Output array with softmax applied and quantized.
        """
        y = nn.softmax(inputs, axis=self.axis)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedLeakyReLU(nn.Module):
    """Quantized Leaky ReLU activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.
    negative_slope : float, default=0.01
        Slope for negative inputs.

    Examples
    --------
    >>> layer = QuantizedLeakyReLU(chop=chop, negative_slope=0.2)
    >>> output = layer(x)
    """

    chop: Optional[Any] = None
    negative_slope: float = 0.01

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized leaky ReLU layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with leaky ReLU applied and quantized.
        """
        y = nn.leaky_relu(inputs, negative_slope=self.negative_slope)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedELU(nn.Module):
    """Quantized ELU activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.
    alpha : float, default=1.0
        Scale for negative inputs.

    Examples
    --------
    >>> layer = QuantizedELU(chop=chop, alpha=1.0)
    >>> output = layer(x)
    """

    chop: Optional[Any] = None
    alpha: float = 1.0

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized ELU layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with ELU applied and quantized.
        """
        y = nn.elu(inputs, alpha=self.alpha)
        if self.chop is not None:
            y = self.chop(y)
        return y


class QuantizedSiLU(nn.Module):
    """Quantized SiLU (Swish) activation for JAX/Flax QAT.

    Parameters
    ----------
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedSiLU(chop=chop)
    >>> output = layer(x)

    Notes
    -----
    SiLU (Sigmoid Linear Unit), also known as Swish, is defined as:
    SiLU(x) = x * sigmoid(x)
    """

    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized SiLU layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with SiLU applied and quantized.
        """
        y = nn.silu(inputs)
        if self.chop is not None:
            y = self.chop(y)
        return y


# ===================================================================
# Quantized Normalization Layers
# ===================================================================

class QuantizedBatchNorm(nn.Module):
    """Quantized Batch Normalization for JAX/Flax QAT.

    Parameters
    ----------
    use_running_average : bool, optional
        If True, use running statistics for normalization (inference mode).
        If False, use batch statistics (training mode).
        If None, determined by `use_running_average` argument in __call__.
    momentum : float, default=0.99
        Momentum for updating running statistics.
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedBatchNorm(chop=chop)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> # Training mode
    >>> output = layer.apply(variables, x, use_running_average=False)
    >>> # Inference mode
    >>> output = layer.apply(variables, x, use_running_average=True)

    Notes
    -----
    Batch normalization normalizes activations across the batch dimension.
    Running statistics (mean, variance) are tracked during training and
    used during inference.
    """

    use_running_average: Optional[bool] = None
    momentum: float = 0.99
    epsilon: float = 1e-5
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Forward pass of the quantized batch normalization layer.

        Parameters
        ----------
        x : jax.Array
            Input array.
        use_running_average : bool, optional
            If True, use running statistics. If False, use batch statistics.
            If None, use self.use_running_average.

        Returns
        -------
        jax.Array
            Normalized and quantized output.
        """
        if use_running_average is None:
            use_running_average = self.use_running_average

        y = nn.BatchNorm(
            use_running_average=use_running_average,
            momentum=self.momentum,
            epsilon=self.epsilon
        )(x)

        if self.chop is not None:
            y = self.chop(y)

        return y


# Aliases for different dimensions (BatchNorm works for all dims in Flax)
QuantizedBatchNorm1d = QuantizedBatchNorm
QuantizedBatchNorm2d = QuantizedBatchNorm
QuantizedBatchNorm3d = QuantizedBatchNorm


class QuantizedLayerNorm(nn.Module):
    """Quantized Layer Normalization for JAX/Flax QAT.

    Parameters
    ----------
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    use_bias : bool, default=True
        Whether to use bias parameter.
    use_scale : bool, default=True
        Whether to use scale (gamma) parameter.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedLayerNorm(chop=chop)
    >>> x = jnp.ones((32, 128))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)

    Notes
    -----
    Layer normalization normalizes activations across the feature dimension.
    Commonly used in transformers and RNNs.
    """

    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, x):
        """Forward pass of the quantized layer normalization layer.

        Parameters
        ----------
        x : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Normalized and quantized output.
        """
        y = nn.LayerNorm(
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale
        )(x)

        if self.chop is not None:
            y = self.chop(y)

        return y


class QuantizedGroupNorm(nn.Module):
    """Quantized Group Normalization for JAX/Flax QAT.

    Parameters
    ----------
    num_groups : int
        Number of groups to divide channels into.
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    use_bias : bool, default=True
        Whether to use bias parameter.
    use_scale : bool, default=True
        Whether to use scale (gamma) parameter.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> # 32 channels divided into 8 groups
    >>> layer = QuantizedGroupNorm(num_groups=8, chop=chop)
    >>> x = jnp.ones((16, 28, 28, 32))  # (batch, height, width, channels)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)

    Notes
    -----
    Group normalization divides channels into groups and normalizes within
    each group independently. It's a middle ground between layer norm and
    instance norm.
    """

    num_groups: int
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, x):
        """Forward pass of the quantized group normalization layer.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (..., channels).

        Returns
        -------
        jax.Array
            Normalized and quantized output.
        """
        y = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale
        )(x)

        if self.chop is not None:
            y = self.chop(y)

        return y


# ===================================================================
# Quantized Regularization Layers
# ===================================================================

class QuantizedDropout(nn.Module):
    """Quantized Dropout for JAX/Flax QAT.

    Parameters
    ----------
    rate : float, default=0.5
        Dropout rate (probability of dropping a unit).
    deterministic : bool, optional
        If True, dropout is disabled (inference mode).
        If False, dropout is applied (training mode).
        If None, determined by `deterministic` argument in __call__.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedDropout(rate=0.5, chop=chop)
    >>> x = jnp.ones((32, 128))
    >>> variables = layer.init(jax.random.PRNGKey(0), x, deterministic=False)
    >>> # Training mode
    >>> output = layer.apply(variables, x, deterministic=False, 
    ...                      rngs={'dropout': jax.random.PRNGKey(1)})
    >>> # Inference mode
    >>> output = layer.apply(variables, x, deterministic=True)

    Notes
    -----
    Dropout randomly zeros elements during training to prevent overfitting.
    During inference, all elements are kept (deterministic=True).
    """

    rate: float = 0.5
    deterministic: Optional[bool] = None
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        """Forward pass of the quantized dropout layer.

        Parameters
        ----------
        inputs : jax.Array
            Input array.
        deterministic : bool, optional
            If True, disable dropout. If False, apply dropout.
            If None, use self.deterministic.

        Returns
        -------
        jax.Array
            Output array with dropout applied and quantized.
        """
        if deterministic is None:
            deterministic = self.deterministic

        if self.chop is not None:
            inputs = self.chop(inputs)

        y = nn.Dropout(rate=self.rate, deterministic=deterministic)(inputs)

        if self.chop is not None:
            y = self.chop(y)

        return y


# ===================================================================
# Quantized Embedding Layer
# ===================================================================

class QuantizedEmbed(nn.Module):
    """Quantized Embedding layer for JAX/Flax QAT.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary (number of unique tokens).
    features : int
        Dimensionality of the embedding vectors.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.
    dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the computation.
    param_dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the embedding table.
    embedding_init : callable, default=uniform()
        Initializer for the embedding table.

    Examples
    --------
    >>> layer = QuantizedEmbed(num_embeddings=1000, features=128, chop=chop)
    >>> indices = jnp.array([1, 5, 10, 42])  # Token indices
    >>> variables = layer.init(jax.random.PRNGKey(0), indices)
    >>> embeddings = layer.apply(variables, indices)
    >>> embeddings.shape  # (4, 128)

    Notes
    -----
    Embedding layer maps discrete token indices to continuous vectors.
    Commonly used as the first layer in NLP models.
    """

    num_embeddings: int
    features: int
    chop: Optional[Any] = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    embedding_init: Callable = nn.initializers.uniform(scale=1.0)

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the quantized embedding layer.

        Parameters
        ----------
        inputs : jax.Array
            Integer array of token indices, shape (...).

        Returns
        -------
        jax.Array
            Embedding vectors, shape (..., features).
        """
        embedding = self.param(
            'embedding',
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype
        )

        if self.chop is not None:
            embedding = self.chop(embedding)

        # Lookup embeddings
        output = embedding[inputs.astype(jnp.int32)]

        if self.chop is not None:
            output = self.chop(output)

        return jnp.asarray(output, self.dtype)


# Alias for consistency with PyTorch naming
QuantizedEmbedding = QuantizedEmbed


# ===================================================================
# Placeholder Classes for Not-Yet-Implemented Layers
# ===================================================================

def _not_implemented_layer(name):
    """Create a placeholder class for layers not yet implemented.

    Parameters
    ----------
    name : str
        Name of the layer.

    Returns
    -------
    type
        A Flax Module class that raises NotImplementedError when called.

    Notes
    -----
    This provides clear error messages for unimplemented layers while
    maintaining API consistency with the PyTorch backend.
    """
    class NotImplementedLayer(nn.Module):
        """Placeholder for {name} - not yet implemented in JAX backend."""
        chop: Optional[Any] = None

        def __call__(self, *args, **kwargs):
            raise NotImplementedError(
                f"{name} is not yet implemented for JAX/Flax backend. "
                f"Contributions are welcome! Please either:\n"
                f"  1. Use the PyTorch backend: pychop.backend('torch')\n"
                f"  2. Implement this layer following the pattern in pychop/jx/layers.py\n"
                f"  3. Open an issue at https://github.com/inEXASCALE/pychop/issues"
            )

    NotImplementedLayer.__name__ = name
    NotImplementedLayer.__doc__ = f"Placeholder for {name} - not yet implemented in JAX backend."
    return NotImplementedLayer



# Transposed convolution (lower priority)
QuantizedConvTranspose1d = _not_implemented_layer("QuantizedConvTranspose1d")
QuantizedConvTranspose2d = _not_implemented_layer("QuantizedConvTranspose2d")
QuantizedConvTranspose3d = _not_implemented_layer("QuantizedConvTranspose3d")

# Instance normalization (less common)
QuantizedInstanceNorm1d = _not_implemented_layer("QuantizedInstanceNorm1d")
QuantizedInstanceNorm2d = _not_implemented_layer("QuantizedInstanceNorm2d")
QuantizedInstanceNorm3d = _not_implemented_layer("QuantizedInstanceNorm3d")

# PReLU (learnable parameter requires careful handling)
# ===================================================================
# Quantized Pooling Layers
# ===================================================================

class QuantizedMaxPool1d(nn.Module):
    """Quantized 1D max pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    window_shape : int
        Size of the pooling window.
    strides : int, optional
        Stride of the pooling operation. If None, defaults to window_shape.
    padding : str, default='VALID'
        Padding mode ('VALID' or 'SAME').
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedMaxPool1d(window_shape=2, chop=chop)
    >>> x = jnp.ones((32, 100, 64))  # (batch, length, channels)
    >>> output = layer(x)
    """

    window_shape: int
    strides: Optional[int] = None
    padding: str = 'VALID'
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 1D max pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, length, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        strides = self.strides if self.strides is not None else self.window_shape
        
        # JAX max pooling
        y = jax.lax.reduce_window(
            inputs,
            -jnp.inf,
            jax.lax.max,
            (1, self.window_shape, 1),
            (1, strides, 1),
            self.padding
        )
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedMaxPool2d(nn.Module):
    """Quantized 2D max pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    strides : int or tuple of int, optional
        Stride of the pooling operation. If None, defaults to window_shape.
    padding : str, default='VALID'
        Padding mode ('VALID' or 'SAME').
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedMaxPool2d(window_shape=2, chop=chop)
    >>> x = jnp.ones((32, 28, 28, 64))  # (batch, height, width, channels)
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int]]
    strides: Optional[Union[int, Tuple[int, int]]] = None
    padding: str = 'VALID'
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 2D max pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, height, width, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        if isinstance(self.window_shape, int):
            window_shape = (1, self.window_shape, self.window_shape, 1)
        else:
            window_shape = (1, self.window_shape[0], self.window_shape[1], 1)
        
        if self.strides is None:
            strides = window_shape
        elif isinstance(self.strides, int):
            strides = (1, self.strides, self.strides, 1)
        else:
            strides = (1, self.strides[0], self.strides[1], 1)
        
        # JAX max pooling
        y = jax.lax.reduce_window(
            inputs,
            -jnp.inf,
            jax.lax.max,
            window_shape,
            strides,
            self.padding
        )
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedMaxPool3d(nn.Module):
    """Quantized 3D max pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    strides : int or tuple of int, optional
        Stride of the pooling operation. If None, defaults to window_shape.
    padding : str, default='VALID'
        Padding mode ('VALID' or 'SAME').
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedMaxPool3d(window_shape=2, chop=chop)
    >>> x = jnp.ones((8, 16, 16, 16, 32))  # (batch, depth, height, width, channels)
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int, int]]
    strides: Optional[Union[int, Tuple[int, int, int]]] = None
    padding: str = 'VALID'
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 3D max pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, depth, height, width, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        if isinstance(self.window_shape, int):
            window_shape = (1, self.window_shape, self.window_shape, self.window_shape, 1)
        else:
            window_shape = (1, self.window_shape[0], self.window_shape[1], self.window_shape[2], 1)
        
        if self.strides is None:
            strides = window_shape
        elif isinstance(self.strides, int):
            strides = (1, self.strides, self.strides, self.strides, 1)
        else:
            strides = (1, self.strides[0], self.strides[1], self.strides[2], 1)
        
        # JAX max pooling
        y = jax.lax.reduce_window(
            inputs,
            -jnp.inf,
            jax.lax.max,
            window_shape,
            strides,
            self.padding
        )
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedAvgPool1d(nn.Module):
    """Quantized 1D average pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    window_shape : int
        Size of the pooling window.
    strides : int, optional
        Stride of the pooling operation. If None, defaults to window_shape.
    padding : str, default='VALID'
        Padding mode ('VALID' or 'SAME').
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedAvgPool1d(window_shape=2, chop=chop)
    >>> x = jnp.ones((32, 100, 64))
    >>> output = layer(x)
    """

    window_shape: int
    strides: Optional[int] = None
    padding: str = 'VALID'
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 1D average pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, length, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        strides = self.strides if self.strides is not None else self.window_shape
        
        # JAX average pooling
        y = jax.lax.reduce_window(
            inputs,
            0.0,
            jax.lax.add,
            (1, self.window_shape, 1),
            (1, strides, 1),
            self.padding
        )
        
        # Divide by window size
        y = y / self.window_shape
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedAvgPool2d(nn.Module):
    """Quantized 2D average pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    strides : int or tuple of int, optional
        Stride of the pooling operation. If None, defaults to window_shape.
    padding : str, default='VALID'
        Padding mode ('VALID' or 'SAME').
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedAvgPool2d(window_shape=2, chop=chop)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int]]
    strides: Optional[Union[int, Tuple[int, int]]] = None
    padding: str = 'VALID'
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 2D average pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, height, width, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        if isinstance(self.window_shape, int):
            window_shape = (1, self.window_shape, self.window_shape, 1)
            window_size = self.window_shape * self.window_shape
        else:
            window_shape = (1, self.window_shape[0], self.window_shape[1], 1)
            window_size = self.window_shape[0] * self.window_shape[1]
        
        if self.strides is None:
            strides = window_shape
        elif isinstance(self.strides, int):
            strides = (1, self.strides, self.strides, 1)
        else:
            strides = (1, self.strides[0], self.strides[1], 1)
        
        # JAX average pooling
        y = jax.lax.reduce_window(
            inputs,
            0.0,
            jax.lax.add,
            window_shape,
            strides,
            self.padding
        )
        
        # Divide by window size
        y = y / window_size
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedAvgPool3d(nn.Module):
    """Quantized 3D average pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    strides : int or tuple of int, optional
        Stride of the pooling operation. If None, defaults to window_shape.
    padding : str, default='VALID'
        Padding mode ('VALID' or 'SAME').
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedAvgPool3d(window_shape=2, chop=chop)
    >>> x = jnp.ones((8, 16, 16, 16, 32))
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int, int]]
    strides: Optional[Union[int, Tuple[int, int, int]]] = None
    padding: str = 'VALID'
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 3D average pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, depth, height, width, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        if isinstance(self.window_shape, int):
            window_shape = (1, self.window_shape, self.window_shape, self.window_shape, 1)
            window_size = self.window_shape ** 3
        else:
            window_shape = (1, self.window_shape[0], self.window_shape[1], self.window_shape[2], 1)
            window_size = self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
        
        if self.strides is None:
            strides = window_shape
        elif isinstance(self.strides, int):
            strides = (1, self.strides, self.strides, self.strides, 1)
        else:
            strides = (1, self.strides[0], self.strides[1], self.strides[2], 1)
        
        # JAX average pooling
        y = jax.lax.reduce_window(
            inputs,
            0.0,
            jax.lax.add,
            window_shape,
            strides,
            self.padding
        )
        
        # Divide by window size
        y = y / window_size
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedAdaptiveAvgPool1d(nn.Module):
    """Quantized 1D adaptive average pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    output_size : int
        Target output size.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedAdaptiveAvgPool1d(output_size=50, chop=chop)
    >>> x = jnp.ones((32, 100, 64))
    >>> output = layer(x)
    >>> output.shape  # (32, 50, 64)
    """

    output_size: int
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 1D adaptive average pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, length, channels).

        Returns
        -------
        jax.Array
            Pooled output of shape (batch, output_size, channels).
        """
        batch_size, input_size, channels = inputs.shape
        
        # Compute stride and window size
        stride = input_size // self.output_size
        window_size = input_size - (self.output_size - 1) * stride
        
        # Use reduce_window for pooling
        y = jax.lax.reduce_window(
            inputs,
            0.0,
            jax.lax.add,
            (1, window_size, 1),
            (1, stride, 1),
            'VALID'
        )
        
        y = y / window_size
        
        # Ensure exact output size
        y = y[:, :self.output_size, :]
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedAdaptiveAvgPool2d(nn.Module):
    """Quantized 2D adaptive average pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    output_size : int or tuple of int
        Target output size.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedAdaptiveAvgPool2d(output_size=(7, 7), chop=chop)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> output = layer(x)
    >>> output.shape  # (32, 7, 7, 64)
    """

    output_size: Union[int, Tuple[int, int]]
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 2D adaptive average pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, height, width, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        if isinstance(self.output_size, int):
            output_h = output_w = self.output_size
        else:
            output_h, output_w = self.output_size
        
        batch_size, input_h, input_w, channels = inputs.shape
        
        # Compute strides and window sizes
        stride_h = input_h // output_h
        stride_w = input_w // output_w
        window_h = input_h - (output_h - 1) * stride_h
        window_w = input_w - (output_w - 1) * stride_w
        
        # Use reduce_window for pooling
        y = jax.lax.reduce_window(
            inputs,
            0.0,
            jax.lax.add,
            (1, window_h, window_w, 1),
            (1, stride_h, stride_w, 1),
            'VALID'
        )
        
        y = y / (window_h * window_w)
        
        # Ensure exact output size
        y = y[:, :output_h, :output_w, :]
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedAdaptiveAvgPool3d(nn.Module):
    """Quantized 3D adaptive average pooling layer for JAX/Flax QAT.

    Parameters
    ----------
    output_size : int or tuple of int
        Target output size.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedAdaptiveAvgPool3d(output_size=(4, 4, 4), chop=chop)
    >>> x = jnp.ones((8, 16, 16, 16, 32))
    >>> output = layer(x)
    >>> output.shape  # (8, 4, 4, 4, 32)
    """

    output_size: Union[int, Tuple[int, int, int]]
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 3D adaptive average pooling.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, depth, height, width, channels).

        Returns
        -------
        jax.Array
            Pooled output.
        """
        if isinstance(self.output_size, int):
            output_d = output_h = output_w = self.output_size
        else:
            output_d, output_h, output_w = self.output_size
        
        batch_size, input_d, input_h, input_w, channels = inputs.shape
        
        # Compute strides and window sizes
        stride_d = input_d // output_d
        stride_h = input_h // output_h
        stride_w = input_w // output_w
        window_d = input_d - (output_d - 1) * stride_d
        window_h = input_h - (output_h - 1) * stride_h
        window_w = input_w - (output_w - 1) * stride_w
        
        # Use reduce_window for pooling
        y = jax.lax.reduce_window(
            inputs,
            0.0,
            jax.lax.add,
            (1, window_d, window_h, window_w, 1),
            (1, stride_d, stride_h, stride_w, 1),
            'VALID'
        )
        
        y = y / (window_d * window_h * window_w)
        
        # Ensure exact output size
        y = y[:, :output_d, :output_h, :output_w, :]
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


# ===================================================================
# Quantized Transposed Convolution Layers
# ===================================================================

class QuantizedConvTranspose1d(nn.Module):
    """Quantized 1D transposed convolution layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int
        Size of the convolution kernel.
    strides : int, default=1
        Stride of the convolution.
    padding : str or int, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedConvTranspose1d(features=64, kernel_size=4, strides=2, chop=chop)
    >>> x = jnp.ones((32, 50, 32))
    >>> output = layer(x)
    >>> output.shape  # (32, 100, 64) with stride=2
    """

    features: int
    kernel_size: int
    strides: int = 1
    padding: str = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 1D transposed convolution.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, length, in_channels).

        Returns
        -------
        jax.Array
            Output array.
        """
        kernel_shape = (self.kernel_size, inputs.shape[-1], self.features)
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            kernel_shape
        )
        
        if self.chop is not None:
            kernel = self.chop(kernel)
        
        # Transpose convolution
        y = jax.lax.conv_transpose(
            inputs,
            kernel,
            strides=(self.strides,),
            padding=self.padding,
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            if self.chop is not None:
                bias = self.chop(bias)
            y = y + bias
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedConvTranspose2d(nn.Module):
    """Quantized 2D transposed convolution layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    strides : int or tuple of int, default=(1, 1)
        Stride of the convolution.
    padding : str, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedConvTranspose2d(features=32, kernel_size=4, strides=2, chop=chop)
    >>> x = jnp.ones((32, 14, 14, 64))
    >>> output = layer(x)
    >>> output.shape  # (32, 28, 28, 32) with stride=2
    """

    features: int
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 2D transposed convolution.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, height, width, in_channels).

        Returns
        -------
        jax.Array
            Output array.
        """
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        
        if isinstance(self.strides, int):
            strides = (self.strides, self.strides)
        else:
            strides = self.strides
        
        kernel_shape = kernel_size + (inputs.shape[-1], self.features)
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            kernel_shape
        )
        
        if self.chop is not None:
            kernel = self.chop(kernel)
        
        # Transpose convolution
        y = jax.lax.conv_transpose(
            inputs,
            kernel,
            strides=strides,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            if self.chop is not None:
                bias = self.chop(bias)
            y = y + bias
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


class QuantizedConvTranspose3d(nn.Module):
    """Quantized 3D transposed convolution layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    strides : int or tuple of int, default=(1, 1, 1)
        Stride of the convolution.
    padding : str, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to add a bias term.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedConvTranspose3d(features=16, kernel_size=4, strides=2, chop=chop)
    >>> x = jnp.ones((8, 8, 8, 8, 32))
    >>> output = layer(x)
    >>> output.shape  # (8, 16, 16, 16, 16) with stride=2
    """

    features: int
    kernel_size: Union[int, Tuple[int, int, int]]
    strides: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    padding: str = 'SAME'
    use_bias: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized 3D transposed convolution.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, depth, height, width, in_channels).

        Returns
        -------
        jax.Array
            Output array.
        """
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        
        if isinstance(self.strides, int):
            strides = (self.strides, self.strides, self.strides)
        else:
            strides = self.strides
        
        kernel_shape = kernel_size + (inputs.shape[-1], self.features)
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            kernel_shape
        )
        
        if self.chop is not None:
            kernel = self.chop(kernel)
        
        # Transpose convolution
        y = jax.lax.conv_transpose(
            inputs,
            kernel,
            strides=strides,
            padding=self.padding,
            dimension_numbers=('NHWDC', 'HWDIO', 'NHWDC')
        )
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            if self.chop is not None:
                bias = self.chop(bias)
            y = y + bias
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


# ===================================================================
# Quantized Instance Normalization
# ===================================================================

class QuantizedInstanceNorm(nn.Module):
    """Base quantized instance normalization for JAX/Flax QAT.

    Parameters
    ----------
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    use_bias : bool, default=True
        Whether to use bias parameter.
    use_scale : bool, default=True
        Whether to use scale (gamma) parameter.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Notes
    -----
    Instance normalization normalizes across spatial dimensions for each
    instance and channel independently.
    """

    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, x):
        """Forward pass of quantized instance normalization.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (batch, ..., channels).

        Returns
        -------
        jax.Array
            Normalized and quantized output.
        """
        # Compute axes for normalization (all except batch and channel)
        reduction_axes = tuple(range(1, x.ndim - 1))
        
        # Compute mean and variance
        mean = jnp.mean(x, axis=reduction_axes, keepdims=True)
        var = jnp.var(x, axis=reduction_axes, keepdims=True)
        
        # Normalize
        y = (x - mean) / jnp.sqrt(var + self.epsilon)
        
        # Apply learnable scale and bias
        num_channels = x.shape[-1]
        
        if self.use_scale:
            scale = self.param('scale', nn.initializers.ones, (num_channels,))
            y = y * scale
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (num_channels,))
            y = y + bias
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


# Aliases for different dimensions
QuantizedInstanceNorm1d = QuantizedInstanceNorm
QuantizedInstanceNorm2d = QuantizedInstanceNorm
QuantizedInstanceNorm3d = QuantizedInstanceNorm


# ===================================================================
# Quantized PReLU
# ===================================================================

class QuantizedPReLU(nn.Module):
    """Quantized Parametric ReLU activation for JAX/Flax QAT.

    Parameters
    ----------
    num_parameters : int, default=1
        Number of learnable parameters. If 1, applies same alpha to all channels.
        If > 1, should match number of channels for per-channel alpha.
    init : float, default=0.25
        Initial value for the learnable slope parameter.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedPReLU(num_parameters=64, chop=chop)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> output = layer(x)

    Notes
    -----
    PReLU applies element-wise: PReLU(x) = max(0, x) + alpha * min(0, x)
    where alpha is a learnable parameter.
    """

    num_parameters: int = 1
    init: float = 0.25
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized PReLU.

        Parameters
        ----------
        inputs : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array with PReLU applied and quantized.
        """
        # Initialize learnable slope parameter
        if self.num_parameters == 1:
            alpha_shape = (1,)
        else:
            alpha_shape = (self.num_parameters,)
        
        alpha = self.param(
            'alpha',
            lambda key, shape: jnp.full(shape, self.init),
            alpha_shape
        )
        
        # Apply PReLU: max(0, x) + alpha * min(0, x)
        positive = jnp.maximum(0, inputs)
        negative = alpha * jnp.minimum(0, inputs)
        y = positive + negative
        
        if self.chop is not None:
            y = self.chop(y)
        
        return y


# ===================================================================
# Quantized Recurrent Layers
# ===================================================================

class QuantizedRNNCell(nn.Module):
    """Quantized RNN cell for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of hidden features.
    activation : callable, default=nn.tanh
        Activation function.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> cell = QuantizedRNNCell(features=128, chop=chop)
    >>> carry = jnp.zeros((32, 128))
    >>> x = jnp.ones((32, 64))
    >>> variables = cell.init(jax.random.PRNGKey(0), carry, x)
    >>> new_carry, y = cell.apply(variables, carry, x)
    """

    features: int
    activation: Callable = nn.tanh
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, carry, inputs):
        """Forward pass of quantized RNN cell.

        Parameters
        ----------
        carry : jax.Array
            Hidden state of shape (batch, features).
        inputs : jax.Array
            Input of shape (batch, input_dim).

        Returns
        -------
        new_carry : jax.Array
            New hidden state.
        output : jax.Array
            Cell output (same as new_carry for RNN).
        """
        # Input-to-hidden
        W_ih = self.param(
            'W_ih',
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], self.features)
        )
        
        # Hidden-to-hidden
        W_hh = self.param(
            'W_hh',
            nn.initializers.lecun_normal(),
            (self.features, self.features)
        )
        
        # Bias
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Quantize weights
        if self.chop is not None:
            W_ih = self.chop(W_ih)
            W_hh = self.chop(W_hh)
            bias = self.chop(bias)
        
        # Compute new hidden state
        new_carry = self.activation(
            jnp.dot(inputs, W_ih) + jnp.dot(carry, W_hh) + bias
        )
        
        # Quantize output
        if self.chop is not None:
            new_carry = self.chop(new_carry)
        
        return new_carry, new_carry


class QuantizedRNN(nn.Module):
    """Quantized RNN layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of hidden features.
    return_sequences : bool, default=True
        Whether to return full sequence or only last output.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedRNN(features=128, chop=chop)
    >>> x = jnp.ones((32, 100, 64))  # (batch, time, features)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    return_sequences: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized RNN.

        Parameters
        ----------
        inputs : jax.Array
            Input sequence of shape (batch, time, input_dim).

        Returns
        -------
        jax.Array
            Output sequence or final output depending on return_sequences.
        """
        batch_size = inputs.shape[0]
        
        # Initialize cell
        cell = QuantizedRNNCell(features=self.features, chop=self.chop)
        
        # Initialize hidden state
        carry = jnp.zeros((batch_size, self.features))
        
        # Process sequence
        def scan_fn(carry, x):
            carry, y = cell(carry, x)
            return carry, y
        
        # Transpose to (time, batch, features) for scanning
        inputs_t = jnp.transpose(inputs, (1, 0, 2))
        
        final_carry, outputs = jax.lax.scan(scan_fn, carry, inputs_t)
        
        # Transpose back to (batch, time, features)
        outputs = jnp.transpose(outputs, (1, 0, 2))
        
        if self.return_sequences:
            return outputs
        else:
            return final_carry


class QuantizedLSTMCell(nn.Module):
    """Quantized LSTM cell for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of hidden features.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> cell = QuantizedLSTMCell(features=128, chop=chop)
    >>> carry = (jnp.zeros((32, 128)), jnp.zeros((32, 128)))  # (h, c)
    >>> x = jnp.ones((32, 64))
    >>> variables = cell.init(jax.random.PRNGKey(0), carry, x)
    >>> new_carry, y = cell.apply(variables, carry, x)
    """

    features: int
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, carry, inputs):
        """Forward pass of quantized LSTM cell.

        Parameters
        ----------
        carry : tuple of jax.Array
            Hidden state (h, c) each of shape (batch, features).
        inputs : jax.Array
            Input of shape (batch, input_dim).

        Returns
        -------
        new_carry : tuple of jax.Array
            New hidden state (h, c).
        output : jax.Array
            Cell output (same as new h).
        """
        h, c = carry
        
        # Combined input-to-hidden and hidden-to-hidden weights
        # Gates: input, forget, cell, output
        W_ih = self.param(
            'W_ih',
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], 4 * self.features)
        )
        
        W_hh = self.param(
            'W_hh',
            nn.initializers.lecun_normal(),
            (self.features, 4 * self.features)
        )
        
        bias = self.param('bias', nn.initializers.zeros, (4 * self.features,))
        
        # Quantize weights
        if self.chop is not None:
            W_ih = self.chop(W_ih)
            W_hh = self.chop(W_hh)
            bias = self.chop(bias)
        
        # Compute gates
        gates = jnp.dot(inputs, W_ih) + jnp.dot(h, W_hh) + bias
        
        # Split gates
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        
        # Apply activations
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = nn.tanh(g)
        o = nn.sigmoid(o)
        
        # Update cell state
        new_c = f * c + i * g
        
        # Compute new hidden state
        new_h = o * nn.tanh(new_c)
        
        # Quantize outputs
        if self.chop is not None:
            new_h = self.chop(new_h)
            new_c = self.chop(new_c)
        
        return (new_h, new_c), new_h


class QuantizedLSTM(nn.Module):
    """Quantized LSTM layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of hidden features.
    return_sequences : bool, default=True
        Whether to return full sequence or only last output.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedLSTM(features=128, chop=chop)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    return_sequences: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized LSTM.

        Parameters
        ----------
        inputs : jax.Array
            Input sequence of shape (batch, time, input_dim).

        Returns
        -------
        jax.Array
            Output sequence or final output depending on return_sequences.
        """
        batch_size = inputs.shape[0]
        
        # Initialize cell
        cell = QuantizedLSTMCell(features=self.features, chop=self.chop)
        
        # Initialize hidden state and cell state
        carry = (
            jnp.zeros((batch_size, self.features)),
            jnp.zeros((batch_size, self.features))
        )
        
        # Process sequence
        def scan_fn(carry, x):
            carry, y = cell(carry, x)
            return carry, y
        
        # Transpose to (time, batch, features) for scanning
        inputs_t = jnp.transpose(inputs, (1, 0, 2))
        
        final_carry, outputs = jax.lax.scan(scan_fn, carry, inputs_t)
        
        # Transpose back to (batch, time, features)
        outputs = jnp.transpose(outputs, (1, 0, 2))
        
        if self.return_sequences:
            return outputs
        else:
            return final_carry[0]  # Return final h


class QuantizedGRUCell(nn.Module):
    """Quantized GRU cell for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of hidden features.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> cell = QuantizedGRUCell(features=128, chop=chop)
    >>> carry = jnp.zeros((32, 128))
    >>> x = jnp.ones((32, 64))
    >>> variables = cell.init(jax.random.PRNGKey(0), carry, x)
    >>> new_carry, y = cell.apply(variables, carry, x)
    """

    features: int
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, carry, inputs):
        """Forward pass of quantized GRU cell.

        Parameters
        ----------
        carry : jax.Array
            Hidden state of shape (batch, features).
        inputs : jax.Array
            Input of shape (batch, input_dim).

        Returns
        -------
        new_carry : jax.Array
            New hidden state.
        output : jax.Array
            Cell output (same as new_carry).
        """
        # Combined weights for reset and update gates
        W_ir = self.param(
            'W_ir',
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], 2 * self.features)
        )
        
        W_hr = self.param(
            'W_hr',
            nn.initializers.lecun_normal(),
            (self.features, 2 * self.features)
        )
        
        bias_ir = self.param('bias_ir', nn.initializers.zeros, (2 * self.features,))
        
        # Quantize weights
        if self.chop is not None:
            W_ir = self.chop(W_ir)
            W_hr = self.chop(W_hr)
            bias_ir = self.chop(bias_ir)
        
        # Compute reset and update gates
        gates = jnp.dot(inputs, W_ir) + jnp.dot(carry, W_hr) + bias_ir
        r, z = jnp.split(gates, 2, axis=-1)
        r = nn.sigmoid(r)
        z = nn.sigmoid(z)
        
        # Candidate hidden state
        W_in = self.param(
            'W_in',
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], self.features)
        )
        
        W_hn = self.param(
            'W_hn',
            nn.initializers.lecun_normal(),
            (self.features, self.features)
        )
        
        bias_in = self.param('bias_in', nn.initializers.zeros, (self.features,))
        
        # Quantize weights
        if self.chop is not None:
            W_in = self.chop(W_in)
            W_hn = self.chop(W_hn)
            bias_in = self.chop(bias_in)
        
        n = nn.tanh(jnp.dot(inputs, W_in) + r * jnp.dot(carry, W_hn) + bias_in)
        
        # Update hidden state
        new_carry = (1 - z) * n + z * carry
        
        # Quantize output
        if self.chop is not None:
            new_carry = self.chop(new_carry)
        
        return new_carry, new_carry


class QuantizedGRU(nn.Module):
    """Quantized GRU layer for JAX/Flax QAT.

    Parameters
    ----------
    features : int
        Number of hidden features.
    return_sequences : bool, default=True
        Whether to return full sequence or only last output.
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedGRU(features=128, chop=chop)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    return_sequences: bool = True
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of quantized GRU.

        Parameters
        ----------
        inputs : jax.Array
            Input sequence of shape (batch, time, input_dim).

        Returns
        -------
        jax.Array
            Output sequence or final output depending on return_sequences.
        """
        batch_size = inputs.shape[0]
        
        # Initialize cell
        cell = QuantizedGRUCell(features=self.features, chop=self.chop)
        
        # Initialize hidden state
        carry = jnp.zeros((batch_size, self.features))
        
        # Process sequence
        def scan_fn(carry, x):
            carry, y = cell(carry, x)
            return carry, y
        
        # Transpose to (time, batch, features) for scanning
        inputs_t = jnp.transpose(inputs, (1, 0, 2))
        
        final_carry, outputs = jax.lax.scan(scan_fn, carry, inputs_t)
        
        # Transpose back to (batch, time, features)
        outputs = jnp.transpose(outputs, (1, 0, 2))
        
        if self.return_sequences:
            return outputs
        else:
            return final_carry


# ===================================================================
# Quantized Multi-Head Attention
# ===================================================================

class QuantizedMultiheadAttention(nn.Module):
    """Quantized multi-head attention for JAX/Flax QAT.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    qkv_features : int, optional
        Dimension of query/key/value. If None, uses input feature dimension.
    out_features : int, optional
        Output dimension. If None, uses input feature dimension.
    dropout_rate : float, default=0.0
        Dropout rate for attention weights.
    deterministic : bool, optional
        Whether to apply dropout (False for training, True for inference).
    chop : Chop, Chopf, Chopi, or None, default=None
        Quantizer instance.

    Examples
    --------
    >>> layer = QuantizedMultiheadAttention(num_heads=8, chop=chop)
    >>> x = jnp.ones((32, 100, 512))  # (batch, seq_len, features)
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x, deterministic=True)
    """

    num_heads: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    chop: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=None):
        """Forward pass of quantized multi-head attention.

        Parameters
        ----------
        inputs : jax.Array
            Input array of shape (batch, seq_len, features).
        mask : jax.Array, optional
            Attention mask of shape (batch, seq_len, seq_len).
        deterministic : bool, optional
            Whether to apply dropout.

        Returns
        -------
        jax.Array
            Output array of shape (batch, seq_len, out_features).
        """
        if deterministic is None:
            deterministic = self.deterministic
        
        features = inputs.shape[-1]
        qkv_features = self.qkv_features or features
        out_features = self.out_features or features
        
        assert qkv_features % self.num_heads == 0, \
            "qkv_features must be divisible by num_heads"
        
        head_dim = qkv_features // self.num_heads
        
        # Linear projections for Q, K, V
        dense_q = QuantizedDense(features=qkv_features, chop=self.chop, name='query')
        dense_k = QuantizedDense(features=qkv_features, chop=self.chop, name='key')
        dense_v = QuantizedDense(features=qkv_features, chop=self.chop, name='value')
        
        query = dense_q(inputs)
        key = dense_k(inputs)
        value = dense_v(inputs)
        
        # Reshape for multi-head attention
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        
        query = query.reshape(batch_size, seq_len, self.num_heads, head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))
        
        # Compute attention scores
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(query.dtype)
        
        # Attention weights: (batch, num_heads, seq_len, seq_len)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
        
        if self.chop is not None:
            attn_weights = self.chop(attn_weights)
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)
        
        # Softmax
        attn_weights = nn.softmax(attn_weights, axis=-1)
        
        # Dropout
        if not deterministic and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=False
            )
        
        # Apply attention to values
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        
        # Transpose back and reshape
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, qkv_features)
        
        # Final linear projection
        output = QuantizedDense(features=out_features, chop=self.chop, name='out')(attn_output)
        
        return output


# Alias
QuantizedAttention = QuantizedMultiheadAttention


# ===================================================================
# Integer Quantized Layers (IQuantized*)
# ===================================================================
# These use ChopiSTE for fake quantization

class IQuantizedDense(nn.Module):
    """Integer quantized dense layer.

    Parameters
    ----------
    features : int
        Number of output features.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    use_bias : bool, default=True
        Whether to use bias.

    Examples
    --------
    >>> layer = IQuantizedDense(features=128, bits=8)
    >>> x = jnp.ones((32, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    bits: int = 8
    symmetric: bool = False
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedDense(
            features=self.features,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)


# Alias
IQuantizedLinear = IQuantizedDense


class IQuantizedConv2d(nn.Module):
    """Integer quantized 2D convolution.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple
        Kernel size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple, default=(1, 1)
        Strides.
    padding : str, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to use bias.

    Examples
    --------
    >>> layer = IQuantizedConv2d(features=32, kernel_size=3, bits=8)
    >>> x = jnp.ones((32, 28, 28, 3))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    kernel_size: Union[int, Tuple[int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedConv2d(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)


# Similar for other IQuantized layers - they all wrap the corresponding
# Quantized* layer with ChopiSTE
class IQuantizedConv1d(nn.Module):
    features: int
    kernel_size: int
    bits: int = 8
    symmetric: bool = False
    strides: int = 1
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedConv1d(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)


class IQuantizedConv3d(nn.Module):
    features: int
    kernel_size: Union[int, Tuple[int, int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedConv3d(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)

# ===================================================================
# Integer Quantized Layers - Complete Implementation
# ===================================================================
# All IQuantized* layers wrap their Quantized* counterparts with ChopiSTE

class IQuantizedConvTranspose1d(nn.Module):
    """Integer quantized 1D transposed convolution.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int
        Kernel size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int, default=1
        Strides.
    padding : str, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to use bias.

    Examples
    --------
    >>> layer = IQuantizedConvTranspose1d(features=64, kernel_size=4, strides=2, bits=8)
    >>> x = jnp.ones((32, 50, 32))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    kernel_size: int
    bits: int = 8
    symmetric: bool = False
    strides: int = 1
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedConvTranspose1d(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)


class IQuantizedConvTranspose2d(nn.Module):
    """Integer quantized 2D transposed convolution.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple of int
        Kernel size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple of int, default=(1, 1)
        Strides.
    padding : str, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to use bias.

    Examples
    --------
    >>> layer = IQuantizedConvTranspose2d(features=32, kernel_size=4, strides=2, bits=8)
    >>> x = jnp.ones((32, 14, 14, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    kernel_size: Union[int, Tuple[int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedConvTranspose2d(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)


class IQuantizedConvTranspose3d(nn.Module):
    """Integer quantized 3D transposed convolution.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : int or tuple of int
        Kernel size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple of int, default=(1, 1, 1)
        Strides.
    padding : str, default='SAME'
        Padding mode.
    use_bias : bool, default=True
        Whether to use bias.

    Examples
    --------
    >>> layer = IQuantizedConvTranspose3d(features=16, kernel_size=4, strides=2, bits=8)
    >>> x = jnp.ones((8, 8, 8, 8, 32))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    kernel_size: Union[int, Tuple[int, int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedConvTranspose3d(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            chop=chop
        )(inputs)


class IQuantizedRNN(nn.Module):
    """Integer quantized RNN layer.

    Parameters
    ----------
    features : int
        Number of hidden features.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    return_sequences : bool, default=True
        Whether to return full sequence or only last output.

    Examples
    --------
    >>> layer = IQuantizedRNN(features=128, bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    bits: int = 8
    symmetric: bool = False
    return_sequences: bool = True

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedRNN(
            features=self.features,
            return_sequences=self.return_sequences,
            chop=chop
        )(inputs)


class IQuantizedLSTM(nn.Module):
    """Integer quantized LSTM layer.

    Parameters
    ----------
    features : int
        Number of hidden features.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    return_sequences : bool, default=True
        Whether to return full sequence or only last output.

    Examples
    --------
    >>> layer = IQuantizedLSTM(features=128, bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    bits: int = 8
    symmetric: bool = False
    return_sequences: bool = True

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedLSTM(
            features=self.features,
            return_sequences=self.return_sequences,
            chop=chop
        )(inputs)


class IQuantizedGRU(nn.Module):
    """Integer quantized GRU layer.

    Parameters
    ----------
    features : int
        Number of hidden features.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    return_sequences : bool, default=True
        Whether to return full sequence or only last output.

    Examples
    --------
    >>> layer = IQuantizedGRU(features=128, bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    features: int
    bits: int = 8
    symmetric: bool = False
    return_sequences: bool = True

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedGRU(
            features=self.features,
            return_sequences=self.return_sequences,
            chop=chop
        )(inputs)


class IQuantizedMaxPool1d(nn.Module):
    """Integer quantized 1D max pooling layer.

    Parameters
    ----------
    window_shape : int
        Size of the pooling window.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int, optional
        Stride of the pooling operation.
    padding : str, default='VALID'
        Padding mode.

    Examples
    --------
    >>> layer = IQuantizedMaxPool1d(window_shape=2, bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> output = layer(x)
    """

    window_shape: int
    bits: int = 8
    symmetric: bool = False
    strides: Optional[int] = None
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedMaxPool1d(
            window_shape=self.window_shape,
            strides=self.strides,
            padding=self.padding,
            chop=chop
        )(inputs)


class IQuantizedMaxPool2d(nn.Module):
    """Integer quantized 2D max pooling layer.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple of int, optional
        Stride of the pooling operation.
    padding : str, default='VALID'
        Padding mode.

    Examples
    --------
    >>> layer = IQuantizedMaxPool2d(window_shape=2, bits=8)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Optional[Union[int, Tuple[int, int]]] = None
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedMaxPool2d(
            window_shape=self.window_shape,
            strides=self.strides,
            padding=self.padding,
            chop=chop
        )(inputs)


class IQuantizedMaxPool3d(nn.Module):
    """Integer quantized 3D max pooling layer.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple of int, optional
        Stride of the pooling operation.
    padding : str, default='VALID'
        Padding mode.

    Examples
    --------
    >>> layer = IQuantizedMaxPool3d(window_shape=2, bits=8)
    >>> x = jnp.ones((8, 16, 16, 16, 32))
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Optional[Union[int, Tuple[int, int, int]]] = None
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedMaxPool3d(
            window_shape=self.window_shape,
            strides=self.strides,
            padding=self.padding,
            chop=chop
        )(inputs)


class IQuantizedAvgPool1d(nn.Module):
    """Integer quantized 1D average pooling layer.

    Parameters
    ----------
    window_shape : int
        Size of the pooling window.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int, optional
        Stride of the pooling operation.
    padding : str, default='VALID'
        Padding mode.

    Examples
    --------
    >>> layer = IQuantizedAvgPool1d(window_shape=2, bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> output = layer(x)
    """

    window_shape: int
    bits: int = 8
    symmetric: bool = False
    strides: Optional[int] = None
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedAvgPool1d(
            window_shape=self.window_shape,
            strides=self.strides,
            padding=self.padding,
            chop=chop
        )(inputs)


class IQuantizedAvgPool2d(nn.Module):
    """Integer quantized 2D average pooling layer.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple of int, optional
        Stride of the pooling operation.
    padding : str, default='VALID'
        Padding mode.

    Examples
    --------
    >>> layer = IQuantizedAvgPool2d(window_shape=2, bits=8)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Optional[Union[int, Tuple[int, int]]] = None
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedAvgPool2d(
            window_shape=self.window_shape,
            strides=self.strides,
            padding=self.padding,
            chop=chop
        )(inputs)


class IQuantizedAvgPool3d(nn.Module):
    """Integer quantized 3D average pooling layer.

    Parameters
    ----------
    window_shape : int or tuple of int
        Size of the pooling window.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    strides : int or tuple of int, optional
        Stride of the pooling operation.
    padding : str, default='VALID'
        Padding mode.

    Examples
    --------
    >>> layer = IQuantizedAvgPool3d(window_shape=2, bits=8)
    >>> x = jnp.ones((8, 16, 16, 16, 32))
    >>> output = layer(x)
    """

    window_shape: Union[int, Tuple[int, int, int]]
    bits: int = 8
    symmetric: bool = False
    strides: Optional[Union[int, Tuple[int, int, int]]] = None
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedAvgPool3d(
            window_shape=self.window_shape,
            strides=self.strides,
            padding=self.padding,
            chop=chop
        )(inputs)


class IQuantizedAdaptiveAvgPool1d(nn.Module):
    """Integer quantized 1D adaptive average pooling layer.

    Parameters
    ----------
    output_size : int
        Target output size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedAdaptiveAvgPool1d(output_size=50, bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> output = layer(x)
    >>> output.shape  # (32, 50, 64)
    """

    output_size: int
    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedAdaptiveAvgPool1d(
            output_size=self.output_size,
            chop=chop
        )(inputs)


class IQuantizedAdaptiveAvgPool2d(nn.Module):
    """Integer quantized 2D adaptive average pooling layer.

    Parameters
    ----------
    output_size : int or tuple of int
        Target output size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedAdaptiveAvgPool2d(output_size=(7, 7), bits=8)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> output = layer(x)
    >>> output.shape  # (32, 7, 7, 64)
    """

    output_size: Union[int, Tuple[int, int]]
    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedAdaptiveAvgPool2d(
            output_size=self.output_size,
            chop=chop
        )(inputs)


class IQuantizedAdaptiveAvgPool3d(nn.Module):
    """Integer quantized 3D adaptive average pooling layer.

    Parameters
    ----------
    output_size : int or tuple of int
        Target output size.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedAdaptiveAvgPool3d(output_size=(4, 4, 4), bits=8)
    >>> x = jnp.ones((8, 16, 16, 16, 32))
    >>> output = layer(x)
    >>> output.shape  # (8, 4, 4, 4, 32)
    """

    output_size: Union[int, Tuple[int, int, int]]
    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedAdaptiveAvgPool3d(
            output_size=self.output_size,
            chop=chop
        )(inputs)


class IQuantizedBatchNorm1d(nn.Module):
    """Integer quantized 1D batch normalization.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    use_running_average : bool, optional
        Whether to use running statistics.
    momentum : float, default=0.99
        Momentum for running statistics.
    epsilon : float, default=1e-5
        Small constant for numerical stability.

    Examples
    --------
    >>> layer = IQuantizedBatchNorm1d(bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x, use_running_average=False)
    """

    bits: int = 8
    symmetric: bool = False
    use_running_average: Optional[bool] = None
    momentum: float = 0.99
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedBatchNorm(
            use_running_average=use_running_average if use_running_average is not None else self.use_running_average,
            momentum=self.momentum,
            epsilon=self.epsilon,
            chop=chop
        )(x, use_running_average)


# Aliases for different dimensions
IQuantizedBatchNorm2d = IQuantizedBatchNorm1d
IQuantizedBatchNorm3d = IQuantizedBatchNorm1d


class IQuantizedLayerNorm(nn.Module):
    """Integer quantized layer normalization.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    use_bias : bool, default=True
        Whether to use bias parameter.
    use_scale : bool, default=True
        Whether to use scale parameter.

    Examples
    --------
    >>> layer = IQuantizedLayerNorm(bits=8)
    >>> x = jnp.ones((32, 128))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    bits: int = 8
    symmetric: bool = False
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedLayerNorm(
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            chop=chop
        )(x)


class IQuantizedGroupNorm(nn.Module):
    """Integer quantized group normalization.

    Parameters
    ----------
    num_groups : int
        Number of groups to divide channels into.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    use_bias : bool, default=True
        Whether to use bias parameter.
    use_scale : bool, default=True
        Whether to use scale parameter.

    Examples
    --------
    >>> layer = IQuantizedGroupNorm(num_groups=8, bits=8)
    >>> x = jnp.ones((16, 28, 28, 32))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    num_groups: int
    bits: int = 8
    symmetric: bool = False
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedGroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            chop=chop
        )(x)


class IQuantizedInstanceNorm1d(nn.Module):
    """Integer quantized 1D instance normalization.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    use_bias : bool, default=True
        Whether to use bias parameter.
    use_scale : bool, default=True
        Whether to use scale parameter.

    Examples
    --------
    >>> layer = IQuantizedInstanceNorm1d(bits=8)
    >>> x = jnp.ones((32, 100, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    bits: int = 8
    symmetric: bool = False
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedInstanceNorm(
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            chop=chop
        )(x)


# Aliases for different dimensions
IQuantizedInstanceNorm2d = IQuantizedInstanceNorm1d
IQuantizedInstanceNorm3d = IQuantizedInstanceNorm1d


class IQuantizedMultiheadAttention(nn.Module):
    """Integer quantized multi-head attention.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    qkv_features : int, optional
        Dimension of query/key/value.
    out_features : int, optional
        Output dimension.
    dropout_rate : float, default=0.0
        Dropout rate for attention weights.
    deterministic : bool, optional
        Whether to apply dropout.

    Examples
    --------
    >>> layer = IQuantizedMultiheadAttention(num_heads=8, bits=8)
    >>> x = jnp.ones((32, 100, 512))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x, deterministic=True)
    """

    num_heads: int
    bits: int = 8
    symmetric: bool = False
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=None):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedMultiheadAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.out_features,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic if deterministic is not None else self.deterministic,
            chop=chop
        )(inputs, mask, deterministic)


# Alias
IQuantizedAttention = IQuantizedMultiheadAttention


class IQuantizedReLU(nn.Module):
    """Integer quantized ReLU activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedReLU(bits=8)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedReLU(chop=chop)(inputs)


class IQuantizedSigmoid(nn.Module):
    """Integer quantized Sigmoid activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedSigmoid(bits=8)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedSigmoid(chop=chop)(inputs)


class IQuantizedTanh(nn.Module):
    """Integer quantized Tanh activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedTanh(bits=8)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedTanh(chop=chop)(inputs)


class IQuantizedLeakyReLU(nn.Module):
    """Integer quantized Leaky ReLU activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    negative_slope : float, default=0.01
        Slope for negative inputs.

    Examples
    --------
    >>> layer = IQuantizedLeakyReLU(bits=8, negative_slope=0.2)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False
    negative_slope: float = 0.01

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedLeakyReLU(chop=chop, negative_slope=self.negative_slope)(inputs)


class IQuantizedELU(nn.Module):
    """Integer quantized ELU activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    alpha : float, default=1.0
        Scale for negative inputs.

    Examples
    --------
    >>> layer = IQuantizedELU(bits=8, alpha=1.0)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False
    alpha: float = 1.0

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedELU(chop=chop, alpha=self.alpha)(inputs)


class IQuantizedPReLU(nn.Module):
    """Integer quantized PReLU activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    num_parameters : int, default=1
        Number of learnable parameters.
    init : float, default=0.25
        Initial value for the learnable slope parameter.

    Examples
    --------
    >>> layer = IQuantizedPReLU(bits=8, num_parameters=64)
    >>> x = jnp.ones((32, 28, 28, 64))
    >>> variables = layer.init(jax.random.PRNGKey(0), x)
    >>> output = layer.apply(variables, x)
    """

    bits: int = 8
    symmetric: bool = False
    num_parameters: int = 1
    init: float = 0.25

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedPReLU(
            chop=chop,
            num_parameters=self.num_parameters,
            init=self.init
        )(inputs)


class IQuantizedSiLU(nn.Module):
    """Integer quantized SiLU (Swish) activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.

    Examples
    --------
    >>> layer = IQuantizedSiLU(bits=8)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedSiLU(chop=chop)(inputs)


class IQuantizedSoftmax(nn.Module):
    """Integer quantized Softmax activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    axis : int, default=-1
        Axis along which to apply softmax.

    Examples
    --------
    >>> layer = IQuantizedSoftmax(bits=8, axis=-1)
    >>> logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> probs = layer(logits)
    """

    bits: int = 8
    symmetric: bool = False
    axis: int = -1

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedSoftmax(chop=chop, axis=self.axis)(inputs)


class IQuantizedGELU(nn.Module):
    """Integer quantized GELU activation.

    Parameters
    ----------
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    approximate : bool, default=False
        Whether to use approximate GELU implementation.

    Examples
    --------
    >>> layer = IQuantizedGELU(bits=8, approximate=False)
    >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    >>> output = layer(x)
    """

    bits: int = 8
    symmetric: bool = False
    approximate: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedGELU(chop=chop, approximate=self.approximate)(inputs)


class IQuantizedDropout(nn.Module):
    """Integer quantized Dropout.

    Parameters
    ----------
    rate : float, default=0.5
        Dropout rate.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    deterministic : bool, optional
        Whether to apply dropout.

    Examples
    --------
    >>> layer = IQuantizedDropout(rate=0.5, bits=8)
    >>> x = jnp.ones((32, 128))
    >>> variables = layer.init(jax.random.PRNGKey(0), x, deterministic=False)
    >>> output = layer.apply(variables, x, deterministic=False,
    ...                      rngs={'dropout': jax.random.PRNGKey(1)})
    """

    rate: float = 0.5
    bits: int = 8
    symmetric: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedDropout(
            rate=self.rate,
            deterministic=deterministic if deterministic is not None else self.deterministic,
            chop=chop
        )(inputs, deterministic)


class IQuantizedEmbedding(nn.Module):
    """Integer quantized Embedding layer.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary.
    features : int
        Dimensionality of the embedding vectors.
    bits : int, default=8
        Number of bits for quantization.
    symmetric : bool, default=False
        Whether to use symmetric quantization.
    dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the computation.
    param_dtype : jax.numpy.dtype, default=jnp.float32
        Dtype of the embedding table.
    embedding_init : callable, default=uniform()
        Initializer for the embedding table.

    Examples
    --------
    >>> layer = IQuantizedEmbedding(num_embeddings=1000, features=128, bits=8)
    >>> indices = jnp.array([1, 5, 10, 42])
    >>> variables = layer.init(jax.random.PRNGKey(0), indices)
    >>> embeddings = layer.apply(variables, indices)
    >>> embeddings.shape  # (4, 128)
    """

    num_embeddings: int
    features: int
    bits: int = 8
    symmetric: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    embedding_init: Callable = nn.initializers.uniform(scale=1.0)

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        chop = ChopiSTE(bits=self.bits, symmetric=self.symmetric)
        return QuantizedEmbed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            chop=chop,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=self.embedding_init
        )(inputs)


# ===================================================================
# Aliases for Consistency
# ===================================================================
IQuantizedEmbed = IQuantizedEmbedding
QuantizedAvgPool = QuantizedAvgPool2d
IQuantizedAvgPool = IQuantizedAvgPool2d
IQuantizedAttention = IQuantizedMultiheadAttention



# ===================================================================
# Module Exports
# ===================================================================

__all__ = [
    # STE wrappers
    'ChopSTE', 'ChopfSTE', 'ChopiSTE',
    
    # Utilities
    'post_quantization',
    
    # Implemented quantized layers
    'QuantizedLinear', 'QuantizedDense',
    'QuantizedConv1d', 'QuantizedConv2d', 'QuantizedConv3d',
    'QuantizedReLU', 'QuantizedSigmoid', 'QuantizedTanh', 'QuantizedGELU',
    'QuantizedSoftmax', 'QuantizedLeakyReLU', 'QuantizedELU', 'QuantizedSiLU',
    'QuantizedBatchNorm', 'QuantizedBatchNorm1d', 'QuantizedBatchNorm2d', 'QuantizedBatchNorm3d',
    'QuantizedLayerNorm', 'QuantizedGroupNorm',
    'QuantizedDropout',
    'QuantizedEmbedding', 'QuantizedEmbed',
    
    # Placeholders (not yet implemented)
    'QuantizedRNN', 'QuantizedLSTM', 'QuantizedGRU',
    'QuantizedMultiheadAttention', 'QuantizedAttention',
    'QuantizedMaxPool1d', 'QuantizedMaxPool2d', 'QuantizedMaxPool3d',
    'QuantizedAvgPool1d', 'QuantizedAvgPool2d', 'QuantizedAvgPool3d',
    'QuantizedAdaptiveAvgPool1d', 'QuantizedAdaptiveAvgPool2d', 'QuantizedAdaptiveAvgPool3d',
    'QuantizedAvgPool',
    'QuantizedConvTranspose1d', 'QuantizedConvTranspose2d', 'QuantizedConvTranspose3d',
    'QuantizedInstanceNorm1d', 'QuantizedInstanceNorm2d', 'QuantizedInstanceNorm3d',
    'QuantizedPReLU',
    
    # Integer quantized (all placeholders for now)
    'IQuantizedLinear', 'IQuantizedDense',
    'IQuantizedConv1d', 'IQuantizedConv2d', 'IQuantizedConv3d',
    'IQuantizedConvTranspose1d', 'IQuantizedConvTranspose2d', 'IQuantizedConvTranspose3d',
    'IQuantizedRNN', 'IQuantizedLSTM', 'IQuantizedGRU',
    'IQuantizedMaxPool1d', 'IQuantizedMaxPool2d', 'IQuantizedMaxPool3d',
    'IQuantizedAvgPool1d', 'IQuantizedAvgPool2d', 'IQuantizedAvgPool3d',
    'IQuantizedAdaptiveAvgPool1d', 'IQuantizedAdaptiveAvgPool2d', 'IQuantizedAdaptiveAvgPool3d',
    'IQuantizedAvgPool',
    'IQuantizedBatchNorm1d', 'IQuantizedBatchNorm2d', 'IQuantizedBatchNorm3d',
    'IQuantizedLayerNorm', 'IQuantizedGroupNorm',
    'IQuantizedInstanceNorm1d', 'IQuantizedInstanceNorm2d', 'IQuantizedInstanceNorm3d',
    'IQuantizedMultiheadAttention', 'IQuantizedAttention',
    'IQuantizedReLU', 'IQuantizedSigmoid', 'IQuantizedTanh', 'IQuantizedLeakyReLU',
    'IQuantizedELU', 'IQuantizedPReLU', 'IQuantizedSiLU', 'IQuantizedSoftmax', 'IQuantizedGELU',
    'IQuantizedDropout',
    'IQuantizedEmbedding', 'IQuantizedEmbed',
]