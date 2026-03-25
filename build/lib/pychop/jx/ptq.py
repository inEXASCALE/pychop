"""
Post-Training Quantization (PTQ) utilities for JAX/Flax.

This module provides comprehensive PTQ methods for JAX backend with full
feature parity to PyTorch backend:

1. post_quantization: Basic weight-only quantization
2. static_post_quantization: Static quantization with calibration
   - Calibration methods: minmax, percentile, kl_divergence, mse
3. dynamic_post_quantization: Dynamic activation quantization
4. mixed_post_quantization: Mixed-precision quantization (W8A16, W4A8, etc.)

Note: JAX doesn't support PyTorch-style forward hooks, so activation quantization
requires manual integration in the model's __call__ method. This module provides
utilities to make that integration easier.

Author: Xinye Chen
Date: 2026-03-25
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import copy
from typing import Optional, Iterable, Dict, Tuple, Any, List, Callable
import jax.tree_util as tree_util
import numpy as np


# ===================================================================
# Helper: Calibration Algorithms
# ===================================================================

def calibrate_minmax(
    stats: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """
    MinMax calibration (identity function).
    
    Parameters
    ----------
    stats : dict
        {layer_name: (min, max)} from calibration data.
    
    Returns
    -------
    dict
        Same as input stats.
    """
    return stats


def calibrate_percentile(
    stats: Dict[str, Tuple[float, float]],
    percentile: float = 99.99
) -> Dict[str, Tuple[float, float]]:
    """
    Percentile-based calibration to reduce outlier impact.
    
    Parameters
    ----------
    stats : dict
        {layer_name: (min, max)} from calibration data.
    percentile : float, default=99.99
        Percentile to clip (e.g., 99.99 = clip top/bottom 0.01%).
    
    Returns
    -------
    dict
        Clipped stats.
    
    Examples
    --------
    >>> stats = {'layer1': (-10.0, 10.0), 'layer2': (-5.0, 5.0)}
    >>> clipped = calibrate_percentile(stats, percentile=99.9)
    >>> # Clips 0.1% outliers: {'layer1': (-9.99, 9.99), ...}
    """
    clipped = {}
    clip_factor = (100 - percentile) / 100
    
    for name, (min_val, max_val) in stats.items():
        range_val = max_val - min_val
        delta = range_val * clip_factor / 2
        clipped[name] = (min_val + delta, max_val - delta)
    
    return clipped


def calibrate_kl_divergence(
    histograms: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
) -> Dict[str, Tuple[float, float]]:
    """
    KL-divergence calibration (TensorRT-style).
    
    Finds optimal clipping threshold that minimizes KL divergence
    between original and quantized distributions.
    
    Parameters
    ----------
    histograms : dict
        {layer_name: (bin_edges, hist_counts)} from calibration data.
    
    Returns
    -------
    dict
        Optimal clipping ranges {layer_name: (min, max)}.
    
    Notes
    -----
    - Computes KL(P || Q) where P is original distribution, Q is quantized
    - Searches for threshold that minimizes information loss
    - More accurate than percentile but computationally expensive
    
    Examples
    --------
    >>> histograms = {
    >>>     'conv1': (jnp.array([...]), jnp.array([...])),
    >>>     'conv2': (jnp.array([...]), jnp.array([...]))
    >>> }
    >>> optimal_ranges = calibrate_kl_divergence(histograms)
    """
    optimal_ranges = {}
    
    for name, (bin_edges, hist_counts) in histograms.items():
        # Convert to numpy for easier manipulation
        edges = np.array(bin_edges)
        counts = np.array(hist_counts)
        
        # Normalize histogram to probability distribution
        total_count = counts.sum()
        if total_count == 0:
            optimal_ranges[name] = (float(edges[0]), float(edges[-1]))
            continue
        
        probs = counts / total_count
        
        # Try different thresholds (from 80% to 100% of range)
        min_kl = float('inf')
        best_threshold_idx = len(edges) - 1
        
        for threshold_idx in range(int(len(edges) * 0.8), len(edges)):
            # Clip and quantize distribution
            clipped_probs = probs.copy()
            clipped_probs[threshold_idx:] = 0
            
            # Re-normalize
            clipped_sum = clipped_probs.sum()
            if clipped_sum > 0:
                clipped_probs = clipped_probs / clipped_sum
            
            # Compute KL divergence: KL(P || Q) = sum(P * log(P / Q))
            epsilon = 1e-10
            kl_div = np.sum(
                probs * np.log((probs + epsilon) / (clipped_probs + epsilon))
            )
            
            if kl_div < min_kl:
                min_kl = kl_div
                best_threshold_idx = threshold_idx
        
        # Get optimal range (symmetric)
        optimal_max = float(edges[best_threshold_idx])
        optimal_min = -optimal_max
        
        optimal_ranges[name] = (optimal_min, optimal_max)
    
    return optimal_ranges


def calibrate_mse(
    activations: Dict[str, List[jnp.ndarray]],
    num_bits: int = 8
) -> Dict[str, Tuple[float, float]]:
    """
    MSE-based calibration (minimize quantization error).
    
    Parameters
    ----------
    activations : dict
        {layer_name: [list of activation arrays]} from calibration data.
    num_bits : int, default=8
        Number of quantization bits.
    
    Returns
    -------
    dict
        Optimal clipping ranges that minimize MSE.
    
    Notes
    -----
    - Grid searches for threshold that minimizes quantization MSE
    - More accurate than minmax but slower than percentile
    - Good balance between accuracy and speed
    
    Examples
    --------
    >>> activations = {
    >>>     'conv1': [jnp.array(...), jnp.array(...)],
    >>>     'conv2': [jnp.array(...), jnp.array(...)]
    >>> }
    >>> optimal_ranges = calibrate_mse(activations, num_bits=8)
    """
    optimal_ranges = {}
    
    for name, act_list in activations.items():
        # Concatenate all activations
        all_acts = jnp.concatenate([a.flatten() for a in act_list])
        
        # Grid search for optimal clipping threshold
        abs_max = float(jnp.max(jnp.abs(all_acts)))
        
        best_mse = float('inf')
        best_threshold = abs_max
        
        # Try different thresholds
        for alpha in np.linspace(0.8, 1.0, 20):
            threshold = alpha * abs_max
            
            # Clip and quantize
            clipped = jnp.clip(all_acts, -threshold, threshold)
            
            # Simulate quantization
            scale = (2 * threshold) / (2 ** num_bits - 1)
            quantized = jnp.round(clipped / scale) * scale
            
            # Compute MSE
            mse = float(jnp.mean((all_acts - quantized) ** 2))
            
            if mse < best_mse:
                best_mse = mse
                best_threshold = threshold
        
        optimal_ranges[name] = (-best_threshold, best_threshold)
    
    return optimal_ranges


# ===================================================================
# Helper: Activation Quantization Wrapper
# ===================================================================

def create_quantized_forward(
    original_forward: Callable,
    chop,
    activation_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    dynamic: bool = True,
    layer_names: Optional[List[str]] = None
) -> Callable:
    """
    Create a quantized forward function for JAX models.
    
    This is a helper to integrate activation quantization into Flax models
    since JAX doesn't support PyTorch-style forward hooks.
    
    Parameters
    ----------
    original_forward : callable
        Original model's __call__ or apply method.
    chop : Chop, Chopf, or Chopi
        Quantizer for activations.
    activation_stats : dict, optional
        {layer_name: (min, max)} for static quantization.
    dynamic : bool, default=True
        Use dynamic (per-batch) or static (calibrated) quantization.
    layer_names : list of str, optional
        Names of layers to quantize. If None, quantizes all intermediate outputs.
    
    Returns
    -------
    callable
        Wrapped forward function with activation quantization.
    
    Examples
    --------
    >>> # In your Flax model:
    >>> class QuantizedCNN(nn.Module):
    >>>     chop: Any
    >>>     activation_stats: Optional[Dict] = None
    >>>     
    >>>     @nn.compact
    >>>     def __call__(self, x, train: bool = False):
    >>>         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    >>>         x = nn.relu(x)
    >>>         
    >>>         # Apply activation quantization
    >>>         if self.activation_stats:
    >>>             min_val, max_val = self.activation_stats.get('conv1', (-1, 1))
    >>>             x = jnp.clip(x, min_val, max_val)
    >>>         x = self.chop.quantize(x)
    >>>         
    >>>         return x
    """
    from pychop.integer import Chopi
    
    def quantized_forward(x, *args, **kwargs):
        # Run original forward
        output = original_forward(x, *args, **kwargs)
        
        # Apply activation quantization
        if dynamic:
            # Dynamic quantization (no calibration needed)
            if isinstance(chop, Chopi):
                q = chop.quantize(output)
                return chop.dequantize(q)
            elif hasattr(chop, 'quantize'):
                return chop.quantize(output)
            else:
                return chop(output)
        else:
            # Static quantization (requires calibration stats)
            if activation_stats is not None:
                # Find appropriate stats (assume single output layer)
                # In practice, you'd need to handle this per-layer
                stats_key = list(activation_stats.keys())[0] if activation_stats else None
                if stats_key:
                    min_val, max_val = activation_stats[stats_key]
                    output = jnp.clip(output, min_val, max_val)
            
            if isinstance(chop, Chopi):
                q = chop.quantize(output)
                return chop.dequantize(q)
            elif hasattr(chop, 'quantize'):
                return chop.quantize(output)
            else:
                return chop(output)
    
    return quantized_forward


# ===================================================================
# 1. Basic Post-Quantization (Weight-Only)
# ===================================================================

def post_quantization(
    model,
    chop,
    eval_mode: bool = True,
    verbose: bool = False
):
    """
    Basic post-training quantization (weight-only) for JAX.
    
    Parameters
    ----------
    model : dict or pytree
        Flax variables dict {'params': ..., 'batch_stats': ...} or params pytree.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Included for API consistency (no effect in JAX).
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    dict or pytree
        Quantized parameters with same structure as input.
    
    Examples
    --------
    >>> from pychop.jx.layers import ChopiSTE
    >>> from pychop.jx.ptq import post_quantization
    >>> 
    >>> chop = ChopiSTE(bits=8, symmetric=True)
    >>> quantized_params = post_quantization(variables, chop, verbose=True)
    """
    from pychop.jx.layers import ChopSTE, ChopfSTE, ChopiSTE
    from pychop.chop import Chop
    from pychop.fixed_point import Chopf
    from pychop.integer import Chopi
    
    def quantize_leaf(path, param):
        path_str = [str(k.key) if hasattr(k, 'key') else str(k) for k in path]
        path_str_joined = '/'.join(path_str)
        
        # Skip batch_stats
        if 'batch_stats' in path_str_joined:
            return param
        
        # Quantize weights and biases
        if any(name in path_str for name in ['kernel', 'weight', 'bias', 'scale']):
            # For PTQ, bypass STE and use base quantization
            if isinstance(chop, ChopSTE):
                quantized = Chop.__call__(chop, param)
            elif isinstance(chop, ChopfSTE):
                quantized = Chopf.quantize(chop, param)
            elif isinstance(chop, ChopiSTE):
                # Integer: quantize then dequantize (fake quantization)
                q = chop.quantize(param)
                quantized = chop.dequantize(q)
            elif isinstance(chop, Chopi):
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
    
    # Handle Flax variable structure
    if isinstance(model, dict) and 'params' in model:
        quantized_params = tree_util.tree_map_with_path(quantize_leaf, model['params'])
        result = {'params': quantized_params}
        
        # Preserve batch_stats if present
        if 'batch_stats' in model:
            result['batch_stats'] = model['batch_stats']
        
        return result
    else:
        # Just params pytree
        return tree_util.tree_map_with_path(quantize_leaf, model)


# ===================================================================
# 2. Static Post-Quantization (Weights + Activations with Calibration)
# ===================================================================

def static_post_quantization(
    params: Any,
    chop,
    calibration_data: Iterable,
    calibration_method: str = 'minmax',
    percentile: float = 99.99,
    verbose: bool = False,
    model_apply_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Static post-training quantization for JAX (with calibration).
    
    Note: Since JAX doesn't support forward hooks, this function:
    1. Quantizes weights
    2. Collects activation statistics (if model_apply_fn provided)
    3. Returns quantized params + calibration metadata
    
    Users should manually apply activation quantization in their model.
    
    Parameters
    ----------
    params : dict or pytree
        Flax variables or params pytree.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    calibration_data : iterable
        Calibration dataset (list of JAX arrays).
    calibration_method : str, default='minmax'
        Calibration algorithm:
        - 'minmax': Simple min/max clipping
        - 'percentile': Percentile-based clipping
        - 'kl_divergence': KL-divergence optimization
        - 'mse': MSE-based optimization
    percentile : float, default=99.99
        Percentile for 'percentile' calibration.
    verbose : bool, default=False
        Print quantization details.
    model_apply_fn : callable, optional
        Model's apply function for collecting activation stats.
        If None, only weight quantization is performed.
    
    Returns
    -------
    dict
        {
            'params': quantized_params,
            'batch_stats': batch_stats (if present),
            'quant_config': {
                'activation_chop': chop,
                'calibration_method': calibration_method,
                'activation_stats': {layer_name: (min, max)},
                'percentile': percentile (if applicable)
            }
        }
    
    Examples
    --------
    >>> from pychop.jx.layers import ChopiSTE
    >>> from pychop.jx.ptq import static_post_quantization
    >>> 
    >>> chop = ChopiSTE(bits=8, symmetric=True)
    >>> 
    >>> # Option 1: Weight-only (no model_apply_fn)
    >>> result = static_post_quantization(
    >>>     variables, chop,
    >>>     calibration_data=calibration_batches,
    >>>     calibration_method='percentile',
    >>>     verbose=True
    >>> )
    >>> 
    >>> # Option 2: Weights + Activation stats (with model_apply_fn)
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
    >>> 
    >>> # Use the quantized model
    >>> output = model.apply(result, x, train=False)
    """
    from pychop.integer import Chopi
    
    # Step 1: Quantize weights
    if verbose:
        print(f"[Static PTQ] Quantizing weights...")
    
    quantized_result = post_quantization(params, chop, verbose=verbose)
    
    # Step 2: Collect activation statistics (if model_apply_fn provided)
    activation_stats = {}
    
    if model_apply_fn is not None:
        if verbose:
            print(f"[Static PTQ] Collecting activation stats (method={calibration_method})...")
        
        stats = {}
        histograms = {}
        activations = {}
        
        # Run calibration (simplified version - collects final output stats)
        # In practice, you'd need to instrument the model to collect per-layer stats
        batch_count = 0
        for batch in calibration_data:
            output = model_apply_fn(quantized_result, batch)
            
            # Collect global stats (for demonstration)
            if 'output' not in stats:
                stats['output'] = [float(jnp.min(output)), float(jnp.max(output))]
            else:
                stats['output'][0] = min(stats['output'][0], float(jnp.min(output)))
                stats['output'][1] = max(stats['output'][1], float(jnp.max(output)))
            
            # Collect histogram for KL-divergence
            if calibration_method == 'kl_divergence':
                if 'output' not in histograms:
                    histograms['output'] = []
                histograms['output'].append(output.flatten())
            
            # Collect raw activations for MSE
            if calibration_method == 'mse':
                if 'output' not in activations:
                    activations['output'] = []
                activations['output'].append(output)
            
            batch_count += 1
        
        if verbose:
            print(f"[Static PTQ] Calibrated with {batch_count} batches")
        
        # Step 3: Apply calibration algorithm
        if calibration_method == 'minmax':
            activation_stats = {k: tuple(v) for k, v in stats.items()}
            if verbose:
                print("[Static PTQ] Using minmax calibration")
        
        elif calibration_method == 'percentile':
            activation_stats = calibrate_percentile(
                {k: tuple(v) for k, v in stats.items()},
                percentile=percentile
            )
            if verbose:
                print(f"[Static PTQ] Using percentile={percentile} calibration")
        
        elif calibration_method == 'kl_divergence':
            # Compute histograms
            hist_data = {}
            for name, data_list in histograms.items():
                all_data = jnp.concatenate(data_list)
                # Convert to numpy for histogram computation
                all_data_np = np.array(all_data)
                counts, edges = np.histogram(all_data_np, bins=2048)
                hist_data[name] = (jnp.array(edges), jnp.array(counts))
            
            activation_stats = calibrate_kl_divergence(hist_data)
            if verbose:
                print("[Static PTQ] Using KL-divergence calibration")
        
        elif calibration_method == 'mse':
            num_bits = getattr(chop, 'bits', 8)
            activation_stats = calibrate_mse(activations, num_bits=num_bits)
            if verbose:
                print("[Static PTQ] Using MSE calibration")
        
        else:
            raise ValueError(f"Unknown calibration_method: {calibration_method}")
        
        if verbose:
            for name, (min_val, max_val) in activation_stats.items():
                print(f"  [A] {name}: range=[{min_val:.6f}, {max_val:.6f}]")
    
    # Step 4: Add calibration metadata
    if isinstance(quantized_result, dict):
        quantized_result['quant_config'] = {
            'activation_chop': chop,
            'calibration_method': calibration_method,
            'activation_stats': activation_stats,
            'percentile': percentile if calibration_method == 'percentile' else None,
        }
    else:
        # Wrap params in dict
        quantized_result = {
            'params': quantized_result,
            'quant_config': {
                'activation_chop': chop,
                'calibration_method': calibration_method,
                'activation_stats': activation_stats,
                'percentile': percentile if calibration_method == 'percentile' else None,
            }
        }
    
    if verbose:
        print("[Static PTQ] Quantization complete!")
        print("[Static PTQ] Note: Apply activation quantization in your model's __call__ method")
        print("[Static PTQ] using the stats in result['quant_config']['activation_stats']")
    
    return quantized_result


# ===================================================================
# 3. Dynamic Post-Quantization (No Calibration)
# ===================================================================

def dynamic_post_quantization(
    params: Any,
    chop,
    eval_mode: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Dynamic post-training quantization for JAX (no calibration).
    
    Quantizes weights only. Activation quantization should be applied
    dynamically in the model's forward pass.
    
    Parameters
    ----------
    params : dict or pytree
        Flax variables or params pytree.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Included for API consistency (no effect in JAX).
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    dict
        {
            'params': quantized_params,
            'batch_stats': batch_stats (if present),
            'quant_config': {
                'activation_chop': chop,
                'dynamic': True
            }
        }
    
    Examples
    --------
    >>> from pychop.jx.layers import ChopiSTE
    >>> from pychop.jx.ptq import dynamic_post_quantization
    >>> 
    >>> chop = ChopiSTE(bits=8, symmetric=True)
    >>> result = dynamic_post_quantization(variables, chop, verbose=True)
    >>> 
    >>> # In your model's __call__ method:
    >>> def __call__(self, x, train=False):
    >>>     x = nn.Conv(...)(x)
    >>>     x = nn.relu(x)
    >>>     
    >>>     # Apply dynamic activation quantization
    >>>     q = self.chop.quantize(x)
    >>>     x = self.chop.dequantize(q)
    >>>     
    >>>     return x
    """
    if verbose:
        print("[Dynamic PTQ] Quantizing weights...")
    
    quantized_result = post_quantization(params, chop, verbose=verbose)
    
    # Add dynamic quantization flag
    if isinstance(quantized_result, dict):
        quantized_result['quant_config'] = {
            'activation_chop': chop,
            'dynamic': True,
        }
    else:
        # Wrap params in dict
        quantized_result = {
            'params': quantized_result,
            'quant_config': {
                'activation_chop': chop,
                'dynamic': True,
            }
        }
    
    if verbose:
        print("[Dynamic PTQ] Quantization complete (dynamic mode)")
        print("[Dynamic PTQ] Note: Apply activation quantization dynamically in your model")
    
    return quantized_result


# ===================================================================
# 4. Mixed-Precision Post-Quantization
# ===================================================================

def mixed_post_quantization(
    params: Any,
    weight_chop,
    activation_chop,
    calibration_data: Optional[Iterable] = None,
    calibration_method: str = 'minmax',
    percentile: float = 99.99,
    verbose: bool = False,
    model_apply_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Mixed-precision post-training quantization for JAX (W8A16, W4A8, etc.).
    
    Uses separate quantizers for weights and activations.
    
    Parameters
    ----------
    params : dict or pytree
        Flax variables or params pytree.
    weight_chop : Chop, Chopf, or Chopi
        Quantizer for weights.
    activation_chop : Chop, Chopf, or Chopi
        Quantizer for activations.
    calibration_data : iterable, optional
        Calibration data for static activation quantization.
    calibration_method : str, default='minmax'
        Calibration algorithm (same as static_post_quantization).
    percentile : float, default=99.99
        Percentile for 'percentile' calibration.
    verbose : bool, default=False
        Print quantization details.
    model_apply_fn : callable, optional
        Model's apply function for collecting activation stats.
    
    Returns
    -------
    dict
        {
            'params': quantized_params,
            'batch_stats': batch_stats (if present),
            'quant_config': {
                'weight_chop': weight_chop,
                'activation_chop': activation_chop,
                'calibration_method': calibration_method,
                'activation_stats': {...} (if calibration_data provided),
                'dynamic': True/False
            }
        }
    
    Examples
    --------
    >>> from pychop.jx.layers import ChopiSTE, ChopSTE
    >>> from pychop.jx.ptq import mixed_post_quantization
    >>> 
    >>> weight_chop = ChopiSTE(bits=8, symmetric=True)
    >>> activation_chop = ChopSTE(exp_bits=5, sig_bits=10)  # FP16
    >>> 
    >>> # Option 1: Dynamic (no calibration)
    >>> result = mixed_post_quantization(
    >>>     variables, weight_chop, activation_chop,
    >>>     verbose=True
    >>> )
    >>> 
    >>> # Option 2: Static (with calibration)
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
    # Step 1: Quantize weights
    w_bits = getattr(weight_chop, 'bits', 'custom')
    a_bits = getattr(activation_chop, 'bits', 'custom')
    
    if verbose:
        print(f"[Mixed PTQ] Configuration: W{w_bits}A{a_bits}")
        print(f"[Mixed PTQ] Quantizing weights...")
    
    quantized_result = post_quantization(params, weight_chop, verbose=verbose)
    
    # Step 2: Handle activation calibration
    activation_stats = {}
    dynamic = calibration_data is None
    
    if calibration_data is not None and model_apply_fn is not None:
        if verbose:
            print(f"[Mixed PTQ] Collecting activation stats (method={calibration_method})...")
        
        # Use same calibration logic as static_post_quantization
        stats = {}
        histograms = {}
        activations = {}
        
        batch_count = 0
        for batch in calibration_data:
            output = model_apply_fn(quantized_result, batch)
            
            if 'output' not in stats:
                stats['output'] = [float(jnp.min(output)), float(jnp.max(output))]
            else:
                stats['output'][0] = min(stats['output'][0], float(jnp.min(output)))
                stats['output'][1] = max(stats['output'][1], float(jnp.max(output)))
            
            if calibration_method == 'kl_divergence':
                if 'output' not in histograms:
                    histograms['output'] = []
                histograms['output'].append(output.flatten())
            
            if calibration_method == 'mse':
                if 'output' not in activations:
                    activations['output'] = []
                activations['output'].append(output)
            
            batch_count += 1
        
        # Apply calibration algorithm
        if calibration_method == 'percentile':
            activation_stats = calibrate_percentile(
                {k: tuple(v) for k, v in stats.items()},
                percentile=percentile
            )
        elif calibration_method == 'kl_divergence':
            hist_data = {}
            for name, data_list in histograms.items():
                all_data = jnp.concatenate(data_list)
                all_data_np = np.array(all_data)
                counts, edges = np.histogram(all_data_np, bins=2048)
                hist_data[name] = (jnp.array(edges), jnp.array(counts))
            activation_stats = calibrate_kl_divergence(hist_data)
        elif calibration_method == 'mse':
            num_bits = getattr(activation_chop, 'bits', 8)
            activation_stats = calibrate_mse(activations, num_bits=num_bits)
        else:
            activation_stats = {k: tuple(v) for k, v in stats.items()}
        
        if verbose:
            print(f"[Mixed PTQ] Calibrated {batch_count} batches")
            for name, (min_val, max_val) in activation_stats.items():
                print(f"  [A] {name}: range=[{min_val:.6f}, {max_val:.6f}]")
    
    # Step 3: Add metadata
    if isinstance(quantized_result, dict):
        quantized_result['quant_config'] = {
            'weight_chop': weight_chop,
            'activation_chop': activation_chop,
            'calibration_method': calibration_method,
            'activation_stats': activation_stats,
            'dynamic': dynamic,
            'percentile': percentile if calibration_method == 'percentile' else None,
        }
    else:
        quantized_result = {
            'params': quantized_result,
            'quant_config': {
                'weight_chop': weight_chop,
                'activation_chop': activation_chop,
                'calibration_method': calibration_method,
                'activation_stats': activation_stats,
                'dynamic': dynamic,
                'percentile': percentile if calibration_method == 'percentile' else None,
            }
        }
    
    if verbose:
        print(f"[Mixed PTQ] W{w_bits}A{a_bits} quantization complete!")
        if dynamic:
            print("[Mixed PTQ] Mode: Dynamic (apply activation quantization in model)")
        else:
            print("[Mixed PTQ] Mode: Static (use activation_stats for calibration)")
    
    return quantized_result


# ===================================================================
# Export All Public APIs
# ===================================================================

__all__ = [
    # PTQ methods
    'post_quantization',
    'static_post_quantization',
    'dynamic_post_quantization',
    'mixed_post_quantization',
    
    # Calibration algorithms
    'calibrate_minmax',
    'calibrate_percentile',
    'calibrate_kl_divergence',
    'calibrate_mse',
    
    # Utilities
    'create_quantized_forward',
]