"""
Post-Training Quantization (PTQ) unified frontend for PyChop.

This module provides a unified interface for PTQ across PyTorch and JAX backends.

Available PTQ Methods:
1. post_quantization: Basic weight-only quantization
2. static_post_quantization: Static quantization with calibration
3. dynamic_post_quantization: Dynamic activation quantization
4. mixed_post_quantization: Mixed-precision quantization

Calibration Algorithms:
- minmax: Simple min/max clipping
- percentile: Percentile-based clipping (99.99% default)
- kl_divergence: KL-divergence optimization (TensorRT-style)
- mse: MSE-based optimization

Author: Xinye Chen
Date: 2026-03-25
"""

import os
from typing import Optional, Iterable, Callable


def _get_backend():
    """Get current backend from environment variable."""
    return os.environ.get("chop_backend", "torch")


def _import_backend_ptq(backend: str):
    """Dynamically import backend-specific PTQ implementations."""
    if backend == "jax":
        try:
            from .jx import ptq as backend_module
        except ImportError as e:
            if 'flax' in str(e) or 'jax' in str(e):
                raise ImportError(
                    "JAX backend requires 'flax' and 'jax' to be installed. "
                    "Install them with: pip install flax jax jaxlib\n"
                    "Or switch to PyTorch backend: pychop.backend('torch')"
                ) from e
            raise
    elif backend == "torch":
        try:
            from .tch import ptq as backend_module
        except ImportError as e:
            if 'torch' in str(e):
                raise ImportError(
                    "PyTorch backend requires 'torch' to be installed. "
                    "Install it with: pip install torch\n"
                    "Or switch to JAX backend: pychop.backend('jax')"
                ) from e
            raise
    else:
        try:
            from .tch import ptq as backend_module
        except ImportError:
            raise ImportError(
                f"Unknown backend '{backend}' and PyTorch backend is not available. "
                f"Valid backends: 'torch', 'jax'"
            )
    
    return backend_module


# ===================================================================
# 1. Basic Post-Quantization (Weight-Only)
# ===================================================================

def post_quantization(model, chop, eval_mode: bool = True, verbose: bool = False):
    """
    Basic post-training quantization (weight-only).
    
    Dispatches to backend-specific implementation.
    
    Parameters
    ----------
    model : torch.nn.Module or dict
        PyTorch model or JAX variables dict.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    model or dict
        Quantized model (PyTorch) or params dict (JAX).
    
    Examples
    --------
    >>> import pychop
    >>> pychop.backend('torch')
    >>> from pychop import Chopi
    >>> from pychop.ptq import post_quantization
    >>> 
    >>> chop = Chopi(bits=8, symmetric=True)
    >>> model_q = post_quantization(model, chop, verbose=True)
    """
    backend = _get_backend()
    module = _import_backend_ptq(backend)
    return module.post_quantization(model, chop, eval_mode, verbose)


# ===================================================================
# 2. Static Post-Quantization (Weights + Activations with Calibration)
# ===================================================================

def static_post_quantization(
    model,
    chop,
    calibration_data: Iterable,
    calibration_method: str = 'minmax',
    percentile: float = 99.99,
    fuse_bn: bool = True,
    eval_mode: bool = True,
    verbose: bool = False,
    model_apply_fn: Optional[Callable] = None,
):
    """
    Static post-training quantization with activation calibration.
    
    Supports multiple calibration algorithms: minmax, percentile, kl_divergence, mse.
    
    Parameters
    ----------
    model : torch.nn.Module or dict
        PyTorch model or JAX variables dict.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    calibration_data : iterable
        Calibration dataset.
    calibration_method : str, default='minmax'
        Calibration algorithm:
        - 'minmax': Simple min/max clipping
        - 'percentile': Percentile-based clipping
        - 'kl_divergence': KL-divergence optimization
        - 'mse': MSE-based optimization
    percentile : float, default=99.99
        Percentile for 'percentile' calibration.
    fuse_bn : bool, default=True
        Fuse Conv+BN layers (PyTorch only).
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.
    model_apply_fn : callable, optional
        Model's apply function for JAX backend to collect activation stats.
        Required for JAX when using advanced calibration methods.
        
        Example for JAX:
        >>> def apply_fn(params, x):
        >>>     return model.apply(params, x, train=False)
    
    Returns
    -------
    model or dict
        Quantized model (PyTorch) or params dict (JAX).
    
    Examples
    --------
    >>> # PyTorch example
    >>> import pychop
    >>> pychop.backend('torch')
    >>> from pychop import Chopi
    >>> from pychop.ptq import static_post_quantization
    >>> 
    >>> chop = Chopi(bits=8, symmetric=True)
    >>> model_q = static_post_quantization(
    >>>     model, chop,
    >>>     calibration_data=train_loader[:100],
    >>>     calibration_method='percentile',
    >>>     percentile=99.9,
    >>>     verbose=True
    >>> )
    >>> 
    >>> # JAX example
    >>> pychop.backend('jax')
    >>> from pychop.jx.layers import ChopiSTE
    >>> 
    >>> chop = ChopiSTE(bits=8, symmetric=True)
    >>> 
    >>> def apply_fn(params, x):
    >>>     return model.apply(params, x, train=False)
    >>> 
    >>> result = static_post_quantization(
    >>>     variables, chop,
    >>>     calibration_data=calibration_batches,
    >>>     calibration_method='kl_divergence',
    >>>     model_apply_fn=apply_fn,
    >>>     verbose=True
    >>> )
    """
    backend = _get_backend()
    module = _import_backend_ptq(backend)
    
    if backend == 'jax':
        # JAX doesn't support fuse_bn, pass model_apply_fn
        return module.static_post_quantization(
            model, chop, calibration_data,
            calibration_method=calibration_method,
            percentile=percentile,
            verbose=verbose,
            model_apply_fn=model_apply_fn,
        )
    else:
        # PyTorch: ignore model_apply_fn (not needed)
        return module.static_post_quantization(
            model, chop, calibration_data,
            calibration_method=calibration_method,
            percentile=percentile,
            fuse_bn=fuse_bn,
            eval_mode=eval_mode,
            verbose=verbose,
        )


# ===================================================================
# 3. Dynamic Post-Quantization (No Calibration)
# ===================================================================

def dynamic_post_quantization(
    model,
    chop,
    eval_mode: bool = True,
    verbose: bool = False,
):
    """
    Dynamic post-training quantization (no calibration needed).
    
    Dispatches to backend-specific implementation.
    
    Parameters
    ----------
    model : torch.nn.Module or dict
        PyTorch model or JAX variables dict.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    model or dict
        Quantized model (PyTorch) or params dict (JAX).
    
    Examples
    --------
    >>> import pychop
    >>> pychop.backend('torch')
    >>> from pychop import Chopi
    >>> from pychop.ptq import dynamic_post_quantization
    >>> 
    >>> chop = Chopi(bits=8, symmetric=True)
    >>> model_q = dynamic_post_quantization(model, chop, verbose=True)
    """
    backend = _get_backend()
    module = _import_backend_ptq(backend)
    return module.dynamic_post_quantization(model, chop, eval_mode, verbose)


# ===================================================================
# 4. Mixed-Precision Post-Quantization
# ===================================================================

def mixed_post_quantization(
    model,
    weight_chop,
    activation_chop,
    calibration_data: Optional[Iterable] = None,
    calibration_method: str = 'minmax',
    percentile: float = 99.99,
    dynamic: bool = True,
    eval_mode: bool = True,
    verbose: bool = False,
    model_apply_fn: Optional[Callable] = None,
):
    """
    Mixed-precision post-training quantization (W8A16, W4A8, etc.).
    
    Uses separate quantizers for weights and activations.
    
    Parameters
    ----------
    model : torch.nn.Module or dict
        PyTorch model or JAX variables dict.
    weight_chop : Chop, Chopf, or Chopi
        Quantizer for weights.
    activation_chop : Chop, Chopf, or Chopi
        Quantizer for activations.
    calibration_data : iterable, optional
        Calibration data for static activation quantization.
    calibration_method : str, default='minmax'
        Calibration algorithm.
    percentile : float, default=99.99
        Percentile for 'percentile' calibration.
    dynamic : bool, default=True
        Use dynamic activation quantization (PyTorch only).
    eval_mode : bool, default=True
        Set model to eval mode (PyTorch only).
    verbose : bool, default=False
        Print quantization details.
    model_apply_fn : callable, optional
        Model's apply function for JAX backend.
    
    Returns
    -------
    model or dict
        Quantized model (PyTorch) or params dict (JAX).
    
    Examples
    --------
    >>> # PyTorch example
    >>> import pychop
    >>> pychop.backend('torch')
    >>> from pychop import Chopi, Chop
    >>> from pychop.ptq import mixed_post_quantization
    >>> 
    >>> weight_chop = Chopi(bits=8, symmetric=True)
    >>> activation_chop = Chop(exp_bits=5, sig_bits=10)  # FP16
    >>> 
    >>> model_q = mixed_post_quantization(
    >>>     model, weight_chop, activation_chop,
    >>>     calibration_data=train_loader[:50],
    >>>     dynamic=False,
    >>>     verbose=True
    >>> )
    >>> 
    >>> # JAX example
    >>> pychop.backend('jax')
    >>> from pychop.jx.layers import ChopiSTE, ChopSTE
    >>> 
    >>> weight_chop = ChopiSTE(bits=8, symmetric=True)
    >>> activation_chop = ChopSTE(exp_bits=5, sig_bits=10)
    >>> 
    >>> def apply_fn(params, x):
    >>>     return model.apply(params, x, train=False)
    >>> 
    >>> result = mixed_post_quantization(
    >>>     variables, weight_chop, activation_chop,
    >>>     calibration_data=calibration_batches,
    >>>     calibration_method='percentile',
    >>>     model_apply_fn=apply_fn,
    >>>     verbose=True
    >>> )
    """
    backend = _get_backend()
    module = _import_backend_ptq(backend)
    
    if backend == 'jax':
        # JAX doesn't use 'dynamic' parameter
        return module.mixed_post_quantization(
            model, weight_chop, activation_chop,
            calibration_data=calibration_data,
            calibration_method=calibration_method,
            percentile=percentile,
            verbose=verbose,
            model_apply_fn=model_apply_fn,
        )
    else:
        # PyTorch: ignore model_apply_fn
        return module.mixed_post_quantization(
            model, weight_chop, activation_chop,
            calibration_data=calibration_data,
            calibration_method=calibration_method,
            dynamic=dynamic,
            eval_mode=eval_mode,
            verbose=verbose,
        )


# ===================================================================
# Export All Public APIs
# ===================================================================

__all__ = [
    'post_quantization',
    'static_post_quantization',
    'dynamic_post_quantization',
    'mixed_post_quantization',
]