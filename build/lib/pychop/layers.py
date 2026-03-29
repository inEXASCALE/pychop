import os

def _get_backend():
    """Get current backend from environment variable."""
    return os.environ.get("chop_backend", "torch")


def _import_backend_layers(backend: str):
    """Dynamically import backend-specific layer implementations.
    
    Raises
    ------
    ImportError
        If the backend's layers module cannot be imported (e.g., missing dependencies).
    """
    if backend == "jax":
        try:
            from .jx import layers as backend_module
        except ImportError as e:
            if 'flax' in str(e):
                raise ImportError(
                    "JAX backend requires 'flax' to be installed. "
                    "Install it with: pip install flax jax jaxlib\n"
                    "Or switch to PyTorch backend: pychop.backend('torch')"
                ) from e
            raise
    elif backend == "torch":
        try:
            from .tch import layers as backend_module
        except ImportError as e:
            if 'torch' in str(e):
                raise ImportError(
                    "PyTorch backend requires 'torch' to be installed. "
                    "Install it with: pip install torch\n"
                    "Or switch to JAX backend: pychop.backend('jax')"
                ) from e
            raise
    else:
        # Default to torch
        try:
            from .tch import layers as backend_module
        except ImportError:
            raise ImportError(
                f"Unknown backend '{backend}' and PyTorch backend is not available. "
                f"Valid backends: 'torch', 'jax'"
            )
    
    return backend_module


# ==================================================================
# Factory Functions with Better Error Messages
# ==================================================================

def _create_layer_factory(layer_name: str):
    """Create a factory function for a specific layer type."""
    def factory(*args, **kwargs):
        backend = _get_backend()
        try:
            module = _import_backend_layers(backend)
        except ImportError as e:
            raise ImportError(
                f"Cannot create {layer_name}: {e}"
            ) from e
        
        layer_class = getattr(module, layer_name, None)
        if layer_class is None:
            raise AttributeError(
                f"{layer_name} is not available in {backend} backend. "
                f"Please check the documentation or try a different backend."
            )
        return layer_class(*args, **kwargs)
    
    factory.__name__ = layer_name
    factory.__doc__ = f"Create a {layer_name} for the current backend."
    return factory


# ==================================================================
# STE Wrappers (frontend - backend agnostic in concept)
# ==================================================================

def ChopSTE(*args, **kwargs):
    """Create a ChopSTE instance for the current backend.
    
    Raises
    ------
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.ChopSTE(*args, **kwargs)


def ChopfSTE(*args, **kwargs):
    """Create a ChopfSTE instance for the current backend.
    
    Raises
    ------
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.ChopfSTE(*args, **kwargs)


def ChopiSTE(*args, **kwargs):
    """Create a ChopiSTE instance for the current backend.
    
    Raises
    ------
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.ChopiSTE(*args, **kwargs)


# ==================================================================
# Post-Quantization (dispatch to backend-specific implementation)
# ==================================================================

def post_quantization(model, chop, eval_mode: bool = True, verbose: bool = False):
    """Post-training quantization (PTQ) wrapper.
    
    Dispatches to backend-specific implementation.
    
    Parameters
    ----------
    model : torch.nn.Module or flax.linen.Module
        Neural network model.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Whether to set model to evaluation mode (PyTorch only).
    verbose : bool, default=False
        Whether to print quantization details.
    
    Returns
    -------
    model
        Quantized model.
    
    Raises
    ------
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.post_quantization(model, chop, eval_mode, verbose)


# ==================================================================
# Static Post-Quantization (dispatch to backend)
# ==================================================================

def static_post_quantization(model, chop, calibration_data,
                             eval_mode: bool = True, verbose: bool = False):
    """Static post-training quantization with activation calibration.

    Quantizes weights/biases AND activations using the same quantizer.
    Uses *calibration_data* to collect per-layer activation min/max,
    then clamps + quantizes activations during inference.

    Dispatches to backend-specific implementation.

    Parameters
    ----------
    model : torch.nn.Module or flax.linen.Module
        Neural network model.
    chop : Chop, Chopf, or Chopi
        Quantizer instance (used for both weights and activations).
    calibration_data : iterable
        Input data for calibration (DataLoader, list of tensors, etc.).
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.

    Returns
    -------
    model or dict
        PyTorch: quantized ``nn.Module`` with static activation hooks.
        JAX: dict with ``params``, ``activation_stats``, ``quantized_apply``.

    Raises
    ------
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.static_post_quantization(
        model, chop, calibration_data, eval_mode, verbose
    )


# ==================================================================
# Dynamic Post-Quantization (dispatch to backend)
# ==================================================================

def dynamic_post_quantization(model, chop,
                              eval_mode: bool = True, verbose: bool = False):
    """Dynamic post-training quantization — no calibration needed.

    Quantizes weights/biases offline using the same quantizer; activations
    are quantized on-the-fly at every inference step.

    Dispatches to backend-specific implementation.

    Parameters
    ----------
    model : torch.nn.Module or flax.linen.Module
        Neural network model.
    chop : Chop, Chopf, or Chopi
        Quantizer instance (used for both weights and activations).
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.

    Returns
    -------
    model or dict
        PyTorch: quantized ``nn.Module`` with dynamic activation hooks.
        JAX: dict with ``params`` and ``dynamic_apply``.

    Raises
    ------
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.dynamic_post_quantization(model, chop, eval_mode, verbose)


# ==================================================================
# Mixed-Precision Post-Quantization (dispatch to backend)
# ==================================================================

def mixed_post_quantization(model, weight_chop, activation_chop,
                            calibration_data=None, dynamic: bool = True,
                            eval_mode: bool = True, verbose: bool = False):
    """Mixed-precision post-training quantization (e.g. W8A8, W4A16).

    Uses **separate quantizers** for weights and activations, enabling
    fine-grained control over precision allocation such as W8A8, W4A8,
    W8A16, etc.

    Dispatches to backend-specific implementation.

    Parameters
    ----------
    model : torch.nn.Module or flax.linen.Module
        Neural network model.
    weight_chop : Chop, Chopf, Chopi, or None
        Quantizer for weights/biases.  ``None`` = keep full-precision.
    activation_chop : Chop, Chopf, Chopi, or None
        Quantizer for activations.  ``None`` = keep full-precision.
    calibration_data : iterable or None
        Required when ``dynamic=False``.  Ignored when ``dynamic=True``.
    dynamic : bool, default=True
        ``True`` = dynamic activation quantization (no calibration).
        ``False`` = static calibration + clamp + quantize.
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.

    Returns
    -------
    model or dict
        PyTorch: quantized ``nn.Module``.
        JAX: dict with ``params`` and ``mixed_apply``.

    Raises
    ------
    ValueError
        If ``dynamic=False`` and ``calibration_data`` is None.
    ImportError
        If the current backend's dependencies are not installed.
    """
    backend = _get_backend()
    module = _import_backend_layers(backend)
    return module.mixed_post_quantization(
        model, weight_chop, activation_chop,
        calibration_data=calibration_data,
        dynamic=dynamic,
        eval_mode=eval_mode,
        verbose=verbose,
    )


    
# ==================================================================
# Layer Factory Functions
# ==================================================================

def _create_layer_factory(layer_name: str):
    """Create a factory function for a specific layer type."""
    def factory(*args, **kwargs):
        backend = _get_backend()
        module = _import_backend_layers(backend)
        layer_class = getattr(module, layer_name)
        return layer_class(*args, **kwargs)
    factory.__name__ = layer_name
    factory.__doc__ = f"Create a {layer_name} for the current backend."
    return factory


# ==================================================================
# Floating-Point Quantized Layers
# ==================================================================

# Convolution layers
QuantizedLinear = _create_layer_factory("QuantizedLinear")
QuantizedConv1d = _create_layer_factory("QuantizedConv1d")
QuantizedConv2d = _create_layer_factory("QuantizedConv2d")
QuantizedConv3d = _create_layer_factory("QuantizedConv3d")
QuantizedConvTranspose1d = _create_layer_factory("QuantizedConvTranspose1d")
QuantizedConvTranspose2d = _create_layer_factory("QuantizedConvTranspose2d")
QuantizedConvTranspose3d = _create_layer_factory("QuantizedConvTranspose3d")

# Recurrent layers
QuantizedRNN = _create_layer_factory("QuantizedRNN")
QuantizedLSTM = _create_layer_factory("QuantizedLSTM")
QuantizedGRU = _create_layer_factory("QuantizedGRU")

# Pooling layers
QuantizedMaxPool1d = _create_layer_factory("QuantizedMaxPool1d")
QuantizedMaxPool2d = _create_layer_factory("QuantizedMaxPool2d")
QuantizedMaxPool3d = _create_layer_factory("QuantizedMaxPool3d")
QuantizedAvgPool1d = _create_layer_factory("QuantizedAvgPool1d")
QuantizedAvgPool2d = _create_layer_factory("QuantizedAvgPool2d")
QuantizedAvgPool3d = _create_layer_factory("QuantizedAvgPool3d")
QuantizedAdaptiveAvgPool2d = _create_layer_factory("QuantizedAdaptiveAvgPool2d")

# Normalization layers
QuantizedBatchNorm1d = _create_layer_factory("QuantizedBatchNorm1d")
QuantizedBatchNorm2d = _create_layer_factory("QuantizedBatchNorm2d")
QuantizedBatchNorm3d = _create_layer_factory("QuantizedBatchNorm3d")
QuantizedLayerNorm = _create_layer_factory("QuantizedLayerNorm")
QuantizedInstanceNorm1d = _create_layer_factory("QuantizedInstanceNorm1d")
QuantizedInstanceNorm2d = _create_layer_factory("QuantizedInstanceNorm2d")
QuantizedInstanceNorm3d = _create_layer_factory("QuantizedInstanceNorm3d")
QuantizedGroupNorm = _create_layer_factory("QuantizedGroupNorm")

# Attention layers
QuantizedMultiheadAttention = _create_layer_factory("QuantizedMultiheadAttention")
QuantizedAttention = _create_layer_factory("QuantizedMultiheadAttention")  # Alias

# Activation layers
QuantizedReLU = _create_layer_factory("QuantizedReLU")
QuantizedSigmoid = _create_layer_factory("QuantizedSigmoid")
QuantizedTanh = _create_layer_factory("QuantizedTanh")
QuantizedLeakyReLU = _create_layer_factory("QuantizedLeakyReLU")
QuantizedSoftmax = _create_layer_factory("QuantizedSoftmax")
QuantizedGELU = _create_layer_factory("QuantizedGELU")
QuantizedELU = _create_layer_factory("QuantizedELU")
QuantizedPReLU = _create_layer_factory("QuantizedPReLU")

# Dropout
QuantizedDropout = _create_layer_factory("QuantizedDropout")

# Embedding
QuantizedEmbedding = _create_layer_factory("QuantizedEmbedding")

# Aliases
QuantizedAvgPool = QuantizedAvgPool2d


# ==================================================================
# Integer Quantized Layers
# ==================================================================

# Convolution layers
IQuantizedLinear = _create_layer_factory("IQuantizedLinear")
IQuantizedConv1d = _create_layer_factory("IQuantizedConv1d")
IQuantizedConv2d = _create_layer_factory("IQuantizedConv2d")
IQuantizedConv3d = _create_layer_factory("IQuantizedConv3d")
IQuantizedConvTranspose1d = _create_layer_factory("IQuantizedConvTranspose1d")
IQuantizedConvTranspose2d = _create_layer_factory("IQuantizedConvTranspose2d")
IQuantizedConvTranspose3d = _create_layer_factory("IQuantizedConvTranspose3d")

# Recurrent layers
IQuantizedRNN = _create_layer_factory("IQuantizedRNN")
IQuantizedLSTM = _create_layer_factory("IQuantizedLSTM")
IQuantizedGRU = _create_layer_factory("IQuantizedGRU")

# Pooling layers
IQuantizedMaxPool1d = _create_layer_factory("IQuantizedMaxPool1d")
IQuantizedMaxPool2d = _create_layer_factory("IQuantizedMaxPool2d")
IQuantizedMaxPool3d = _create_layer_factory("IQuantizedMaxPool3d")
IQuantizedAvgPool1d = _create_layer_factory("IQuantizedAvgPool1d")
IQuantizedAvgPool2d = _create_layer_factory("IQuantizedAvgPool2d")
IQuantizedAvgPool3d = _create_layer_factory("IQuantizedAvgPool3d")
IQuantizedAdaptiveAvgPool1d = _create_layer_factory("IQuantizedAdaptiveAvgPool1d")
IQuantizedAdaptiveAvgPool2d = _create_layer_factory("IQuantizedAdaptiveAvgPool2d")
IQuantizedAdaptiveAvgPool3d = _create_layer_factory("IQuantizedAdaptiveAvgPool3d")

# Normalization layers
IQuantizedBatchNorm1d = _create_layer_factory("IQuantizedBatchNorm1d")
IQuantizedBatchNorm2d = _create_layer_factory("IQuantizedBatchNorm2d")
IQuantizedBatchNorm3d = _create_layer_factory("IQuantizedBatchNorm3d")
IQuantizedLayerNorm = _create_layer_factory("IQuantizedLayerNorm")
IQuantizedInstanceNorm1d = _create_layer_factory("IQuantizedInstanceNorm1d")
IQuantizedInstanceNorm2d = _create_layer_factory("IQuantizedInstanceNorm2d")
IQuantizedInstanceNorm3d = _create_layer_factory("IQuantizedInstanceNorm3d")
IQuantizedGroupNorm = _create_layer_factory("IQuantizedGroupNorm")

# Attention layers
IQuantizedMultiheadAttention = _create_layer_factory("IQuantizedMultiheadAttention")
IQuantizedAttention = _create_layer_factory("IQuantizedMultiheadAttention")  # Alias

# Activation layers
IQuantizedReLU = _create_layer_factory("IQuantizedReLU")
IQuantizedSigmoid = _create_layer_factory("IQuantizedSigmoid")
IQuantizedTanh = _create_layer_factory("IQuantizedTanh")
IQuantizedLeakyReLU = _create_layer_factory("IQuantizedLeakyReLU")
IQuantizedSoftmax = _create_layer_factory("IQuantizedSoftmax")
IQuantizedGELU = _create_layer_factory("IQuantizedGELU")
IQuantizedELU = _create_layer_factory("IQuantizedELU")
IQuantizedPReLU = _create_layer_factory("IQuantizedPReLU")
IQuantizedSiLU = _create_layer_factory("IQuantizedSiLU")

# Dropout
IQuantizedDropout = _create_layer_factory("IQuantizedDropout")

# Embedding
IQuantizedEmbedding = _create_layer_factory("IQuantizedEmbedding")

# Aliases
IQuantizedAvgPool = IQuantizedAvgPool2d


# ==================================================================
# Export all symbols
# ==================================================================

__all__ = [
    # STE wrappers
    "ChopSTE", "ChopfSTE", "ChopiSTE",
    
    # Utilities
    "post_quantization",
    "static_post_quantization",
    "dynamic_post_quantization",
    "mixed_post_quantization",
    
    # Floating-point quantized layers
    "QuantizedLinear", "QuantizedConv1d", "QuantizedConv2d", "QuantizedConv3d",
    "QuantizedConvTranspose1d", "QuantizedConvTranspose2d", "QuantizedConvTranspose3d",
    "QuantizedRNN", "QuantizedLSTM", "QuantizedGRU",
    "QuantizedMaxPool1d", "QuantizedMaxPool2d", "QuantizedMaxPool3d",
    "QuantizedAvgPool1d", "QuantizedAvgPool2d", "QuantizedAvgPool3d",
    "QuantizedAdaptiveAvgPool2d",
    "QuantizedBatchNorm1d", "QuantizedBatchNorm2d", "QuantizedBatchNorm3d",
    "QuantizedLayerNorm", "QuantizedInstanceNorm1d", "QuantizedInstanceNorm2d",
    "QuantizedInstanceNorm3d", "QuantizedGroupNorm",
    "QuantizedMultiheadAttention", "QuantizedAttention",
    "QuantizedReLU", "QuantizedSigmoid", "QuantizedTanh", "QuantizedLeakyReLU",
    "QuantizedSoftmax", "QuantizedGELU", "QuantizedELU", "QuantizedPReLU",
    "QuantizedDropout", "QuantizedEmbedding", "QuantizedAvgPool",
    
    # Integer quantized layers
    "IQuantizedLinear", "IQuantizedConv1d", "IQuantizedConv2d", "IQuantizedConv3d",
    "IQuantizedConvTranspose1d", "IQuantizedConvTranspose2d", "IQuantizedConvTranspose3d",
    "IQuantizedRNN", "IQuantizedLSTM", "IQuantizedGRU",
    "IQuantizedMaxPool1d", "IQuantizedMaxPool2d", "IQuantizedMaxPool3d",
    "IQuantizedAvgPool1d", "IQuantizedAvgPool2d", "IQuantizedAvgPool3d",
    "IQuantizedAdaptiveAvgPool1d", "IQuantizedAdaptiveAvgPool2d", "IQuantizedAdaptiveAvgPool3d",
    "IQuantizedBatchNorm1d", "IQuantizedBatchNorm2d", "IQuantizedBatchNorm3d",
    "IQuantizedLayerNorm", "IQuantizedInstanceNorm1d", "IQuantizedInstanceNorm2d",
    "IQuantizedInstanceNorm3d", "IQuantizedGroupNorm",
    "IQuantizedMultiheadAttention", "IQuantizedAttention",
    "IQuantizedReLU", "IQuantizedSigmoid", "IQuantizedTanh", "IQuantizedLeakyReLU",
    "IQuantizedSoftmax", "IQuantizedGELU", "IQuantizedELU", "IQuantizedPReLU", "IQuantizedSiLU",
    "IQuantizedDropout", "IQuantizedEmbedding", "IQuantizedAvgPool",
]