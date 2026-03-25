"""
Post-Training Quantization (PTQ) utilities for PyTorch.

This module provides comprehensive PTQ methods with advanced calibration algorithms:
1. post_quantization: Basic weight-only quantization
2. static_post_quantization: Static quantization with calibration (minmax/percentile/KL/MSE)
3. dynamic_post_quantization: Dynamic activation quantization
4. mixed_post_quantization: Mixed-precision quantization (W8A16, W4A8, etc.)

Author: Xinye Chen
Date: 2026-03-25
"""

import torch
import torch.nn as nn
import copy
from typing import Optional, Iterable, Dict, Tuple, List
import numpy as np


# ===================================================================
# Helper Functions
# ===================================================================

def fuse_conv_bn(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    Fuse Conv2d + BatchNorm2d layers for better PTQ accuracy.
    
    Parameters
    ----------
    model : nn.Module
        Model to fuse (modified in-place).
    verbose : bool
        Print fusion details.
    
    Returns
    -------
    nn.Module
        Model with fused layers.
    """
    model.eval()
    
    # Find Conv+BN pairs
    modules_list = list(model.named_modules())
    fused_count = 0
    
    for i, (name1, module1) in enumerate(modules_list):
        if isinstance(module1, nn.Conv2d):
            if i + 1 < len(modules_list):
                name2, module2 = modules_list[i + 1]
                if isinstance(module2, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    # Check channel compatibility
                    if module1.out_channels == module2.num_features:
                        try:
                            # PyTorch built-in fusion
                            from torch.nn.utils.fusion import fuse_conv_bn_eval
                            fused_conv = fuse_conv_bn_eval(module1, module2)
                            
                            # Replace Conv
                            parent_name = '.'.join(name1.split('.')[:-1])
                            conv_attr = name1.split('.')[-1]
                            bn_attr = name2.split('.')[-1]
                            
                            if parent_name:
                                parent = dict(model.named_modules())[parent_name]
                            else:
                                parent = model
                            
                            setattr(parent, conv_attr, fused_conv)
                            setattr(parent, bn_attr, nn.Identity())
                            
                            fused_count += 1
                            if verbose:
                                print(f"[Fused] {name1} + {name2}")
                        except Exception as e:
                            if verbose:
                                print(f"[Skip] Could not fuse {name1} + {name2}: {e}")
    
    if verbose:
        print(f"[Fusion] Total fused layers: {fused_count}")
    
    return model


def calibrate_minmax(
    stats: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """MinMax calibration (identity function)."""
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
    percentile : float
        Percentile to clip (e.g., 99.99 = clip top/bottom 0.01%).
    
    Returns
    -------
    dict
        Clipped stats.
    """
    clipped = {}
    clip_factor = (100 - percentile) / 100
    
    for name, (min_val, max_val) in stats.items():
        range_val = max_val - min_val
        delta = range_val * clip_factor / 2
        clipped[name] = (min_val + delta, max_val - delta)
    
    return clipped


def calibrate_kl_divergence(
    histograms: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
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
    """
    optimal_ranges = {}
    
    for name, (bin_edges, hist_counts) in histograms.items():
        # Convert to numpy for easier manipulation
        edges = bin_edges.cpu().numpy()
        counts = hist_counts.cpu().numpy()
        
        # Normalize histogram to probability distribution
        total_count = counts.sum()
        if total_count == 0:
            optimal_ranges[name] = (edges[0], edges[-1])
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
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-10
            kl_div = np.sum(
                probs * np.log((probs + epsilon) / (clipped_probs + epsilon))
            )
            
            if kl_div < min_kl:
                min_kl = kl_div
                best_threshold_idx = threshold_idx
        
        # Get optimal range
        optimal_max = edges[best_threshold_idx]
        optimal_min = -optimal_max  # Symmetric
        
        optimal_ranges[name] = (optimal_min, optimal_max)
    
    return optimal_ranges


def calibrate_mse(
    activations: Dict[str, List[torch.Tensor]],
    num_bits: int = 8
) -> Dict[str, Tuple[float, float]]:
    """
    MSE-based calibration (minimize quantization error).
    
    Parameters
    ----------
    activations : dict
        {layer_name: [list of activation tensors]} from calibration data.
    num_bits : int
        Number of quantization bits.
    
    Returns
    -------
    dict
        Optimal clipping ranges.
    """
    optimal_ranges = {}
    
    for name, act_list in activations.items():
        # Concatenate all activations
        all_acts = torch.cat([a.flatten() for a in act_list])
        
        # Grid search for optimal clipping threshold
        abs_max = all_acts.abs().max().item()
        
        best_mse = float('inf')
        best_threshold = abs_max
        
        # Try different thresholds
        for alpha in np.linspace(0.8, 1.0, 20):
            threshold = alpha * abs_max
            
            # Clip and quantize
            clipped = torch.clamp(all_acts, -threshold, threshold)
            
            # Simulate quantization
            scale = (2 * threshold) / (2 ** num_bits - 1)
            quantized = torch.round(clipped / scale) * scale
            
            # Compute MSE
            mse = ((all_acts - quantized) ** 2).mean().item()
            
            if mse < best_mse:
                best_mse = mse
                best_threshold = threshold
        
        optimal_ranges[name] = (-best_threshold, best_threshold)
    
    return optimal_ranges


# ===================================================================
# 1. Basic Post-Quantization (Weight-Only)
# ===================================================================

def post_quantization(
    model: torch.nn.Module,
    chop,
    eval_mode: bool = True,
    verbose: bool = False
) -> torch.nn.Module:
    """
    Basic post-training quantization (weight-only).
    
    Quantizes only weights and biases while preserving activations in FP32.
    
    Parameters
    ----------
    model : torch.nn.Module
        Original PyTorch model (remains unmodified).
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Set model to eval mode.
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    torch.nn.Module
        Quantized model (weight-only).
    
    Examples
    --------
    >>> from pychop import Chopi
    >>> from pychop.tch.ptq import post_quantization
    >>> 
    >>> chop = Chopi(bits=8, symmetric=True)
    >>> model_q = post_quantization(model, chop, verbose=True)
    """
    from pychop.integer import Chopi
    
    quantized_model = copy.deepcopy(model)
    
    if eval_mode:
        quantized_model.eval()
    
    device = next(model.parameters()).device
    state_dict = quantized_model.state_dict()
    
    for key in state_dict.keys():
        tensor = state_dict[key].to(device)
        
        if 'weight' in key or 'bias' in key:
            # Use fake quantization for integer types
            if isinstance(chop, Chopi):
                q = chop.quantize(tensor)
                quantized_tensor = chop.dequantize(q)
            elif hasattr(chop, 'quantize'):
                quantized_tensor = chop.quantize(tensor)
            else:
                quantized_tensor = chop(tensor)
        else:
            quantized_tensor = tensor
        
        if quantized_tensor.shape != tensor.shape:
            raise ValueError(f"Shape mismatch for {key}")
        
        state_dict[key] = quantized_tensor
        
        if verbose and ('weight' in key or 'bias' in key):
            print(f"[Quantized] {key}: {quantized_tensor.shape}")
    
    quantized_model.load_state_dict(state_dict)
    return quantized_model


# ===================================================================
# 2. Static Post-Quantization (Weights + Activations with Calibration)
# ===================================================================

def static_post_quantization(
    model: torch.nn.Module,
    chop,
    calibration_data: Iterable,
    calibration_method: str = 'minmax',
    percentile: float = 99.99,
    fuse_bn: bool = True,
    eval_mode: bool = True,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Static post-training quantization with activation calibration.
    
    Quantizes weights and activations using calibration data. Supports
    multiple calibration algorithms: minmax, percentile, KL-divergence, MSE.
    
    Parameters
    ----------
    model : torch.nn.Module
        Original PyTorch model.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    calibration_data : iterable
        Calibration dataset (DataLoader or list of tensors).
    calibration_method : str, default='minmax'
        Calibration algorithm:
        - 'minmax': Simple min/max clipping
        - 'percentile': Percentile-based clipping
        - 'kl_divergence': KL-divergence optimization
        - 'mse': MSE-based optimization
    percentile : float, default=99.99
        Percentile for 'percentile' calibration.
    fuse_bn : bool, default=True
        Fuse Conv+BN layers before quantization.
    eval_mode : bool, default=True
        Set model to eval mode.
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    torch.nn.Module
        Quantized model with static activation hooks.
    
    Examples
    --------
    >>> from pychop import Chopi
    >>> from pychop.tch.ptq import static_post_quantization
    >>> 
    >>> chop = Chopi(bits=8, symmetric=True)
    >>> model_q = static_post_quantization(
    >>>     model, chop,
    >>>     calibration_data=train_loader[:100],
    >>>     calibration_method='percentile',
    >>>     percentile=99.9,
    >>>     verbose=True
    >>> )
    """
    from pychop.integer import Chopi
    
    # Step 1: Deep copy and prepare model
    q_model = copy.deepcopy(model)
    
    if eval_mode:
        q_model.eval()
    
    # Step 2: Fuse Conv+BN (optional)
    if fuse_bn:
        if verbose:
            print("[Static PTQ] Fusing Conv+BN layers...")
        q_model = fuse_conv_bn(q_model, verbose=verbose)
    
    # Step 3: Quantize weights
    if verbose:
        print("[Static PTQ] Quantizing weights...")
    
    with torch.no_grad():
        state_dict = q_model.state_dict()
        for key in state_dict.keys():
            if 'weight' in key or 'bias' in key:
                if isinstance(chop, Chopi):
                    q = chop.quantize(state_dict[key])
                    state_dict[key] = chop.dequantize(q)
                elif hasattr(chop, 'quantize'):
                    state_dict[key] = chop.quantize(state_dict[key])
                else:
                    state_dict[key] = chop(state_dict[key])
                
                if verbose:
                    print(f"  [W] {key}: {state_dict[key].shape}")
        
        q_model.load_state_dict(state_dict)
    
    # Step 4: Collect activation statistics
    if verbose:
        print(f"[Static PTQ] Collecting activation stats (method={calibration_method})...")
    
    stats = {}
    histograms = {}
    activations = {}
    handles = []
    
    def make_calibration_hook(name):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            
            output_data = output.detach()
            
            # Collect min/max
            if name not in stats:
                stats[name] = [output_data.min().item(), output_data.max().item()]
            else:
                stats[name][0] = min(stats[name][0], output_data.min().item())
                stats[name][1] = max(stats[name][1], output_data.max().item())
            
            # Collect histogram for KL-divergence
            if calibration_method == 'kl_divergence':
                if name not in histograms:
                    histograms[name] = []
                hist_data = output_data.flatten().cpu()
                histograms[name].append(hist_data)
            
            # Collect raw activations for MSE
            if calibration_method == 'mse':
                if name not in activations:
                    activations[name] = []
                activations[name].append(output_data.cpu())
        
        return hook
    
    # Register hooks
    target_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, 
                     nn.ReLU, nn.GELU, nn.SiLU)
    
    for name, module in q_model.named_modules():
        if isinstance(module, target_layers):
            handle = module.register_forward_hook(make_calibration_hook(name))
            handles.append(handle)
    
    # Run calibration
    device = next(q_model.parameters()).device
    batch_count = 0
    
    with torch.no_grad():
        for batch in calibration_data:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            q_model(inputs)
            batch_count += 1
    
    # Remove calibration hooks
    for handle in handles:
        handle.remove()
    
    if verbose:
        print(f"[Static PTQ] Calibrated {len(stats)} layers with {batch_count} batches")
    
    # Step 5: Apply calibration algorithm
    if calibration_method == 'minmax':
        final_stats = {k: tuple(v) for k, v in stats.items()}
        if verbose:
            print("[Static PTQ] Using minmax calibration")
    
    elif calibration_method == 'percentile':
        final_stats = calibrate_percentile(
            {k: tuple(v) for k, v in stats.items()},
            percentile=percentile
        )
        if verbose:
            print(f"[Static PTQ] Using percentile={percentile} calibration")
    
    elif calibration_method == 'kl_divergence':
        # Compute histograms
        hist_data = {}
        for name, data_list in histograms.items():
            all_data = torch.cat(data_list)
            counts, edges = torch.histogram(all_data, bins=2048)
            hist_data[name] = (edges, counts)
        
        final_stats = calibrate_kl_divergence(hist_data)
        if verbose:
            print("[Static PTQ] Using KL-divergence calibration")
    
    elif calibration_method == 'mse':
        num_bits = getattr(chop, 'bits', 8)
        final_stats = calibrate_mse(activations, num_bits=num_bits)
        if verbose:
            print("[Static PTQ] Using MSE calibration")
    
    else:
        raise ValueError(f"Unknown calibration_method: {calibration_method}")
    
    # Step 6: Register static quantization hooks
    if verbose:
        print("[Static PTQ] Registering static quantization hooks...")
    
    for name, module in q_model.named_modules():
        if name in final_stats:
            min_val, max_val = final_stats[name]
            
            def make_quant_hook(min_v, max_v):
                def hook(module, input, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    
                    clamped = torch.clamp(output, min_v, max_v)
                    
                    if isinstance(chop, Chopi):
                        q = chop.quantize(clamped)
                        return chop.dequantize(q)
                    elif hasattr(chop, 'quantize'):
                        return chop.quantize(clamped)
                    else:
                        return chop(clamped)
                
                return hook
            
            module.register_forward_hook(make_quant_hook(min_val, max_val))
            
            if verbose:
                print(f"  [A] {name}: range=[{min_val:.6f}, {max_val:.6f}]")
    
    if verbose:
        print("[Static PTQ] Quantization complete!")
    
    return q_model


# ===================================================================
# 3. Dynamic Post-Quantization (No Calibration)
# ===================================================================

def dynamic_post_quantization(
    model: torch.nn.Module,
    chop,
    eval_mode: bool = True,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Dynamic post-training quantization (no calibration needed).
    
    Quantizes weights offline and activations dynamically during inference.
    
    Parameters
    ----------
    model : torch.nn.Module
        Original PyTorch model.
    chop : Chop, Chopf, or Chopi
        Quantizer instance.
    eval_mode : bool, default=True
        Set model to eval mode.
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    torch.nn.Module
        Quantized model with dynamic activation hooks.
    
    Examples
    --------
    >>> from pychop import Chopi
    >>> from pychop.tch.ptq import dynamic_post_quantization
    >>> 
    >>> chop = Chopi(bits=8, symmetric=True)
    >>> model_q = dynamic_post_quantization(model, chop, verbose=True)
    """
    from pychop.integer import Chopi
    
    # Step 1: Quantize weights
    q_model = post_quantization(model, chop, eval_mode=eval_mode, verbose=verbose)
    
    # Step 2: Register dynamic quantization hooks
    if verbose:
        print("[Dynamic PTQ] Registering dynamic quantization hooks...")
    
    target_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear,
                     nn.ReLU, nn.GELU, nn.SiLU)
    
    hook_count = 0
    for name, module in q_model.named_modules():
        if isinstance(module, target_layers):
            def dynamic_hook(module, input, output):
                if not isinstance(output, torch.Tensor):
                    return output
                
                if isinstance(chop, Chopi):
                    q = chop.quantize(output)
                    return chop.dequantize(q)
                elif hasattr(chop, 'quantize'):
                    return chop.quantize(output)
                else:
                    return chop(output)
            
            module.register_forward_hook(dynamic_hook)
            hook_count += 1
            
            if verbose:
                print(f"  [A] {name}")
    
    if verbose:
        print(f"[Dynamic PTQ] Total dynamic hooks: {hook_count}")
    
    return q_model


# ===================================================================
# 4. Mixed-Precision Post-Quantization
# ===================================================================

def mixed_post_quantization(
    model: torch.nn.Module,
    weight_chop,
    activation_chop,
    calibration_data: Optional[Iterable] = None,
    calibration_method: str = 'minmax',
    dynamic: bool = True,
    eval_mode: bool = True,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Mixed-precision post-training quantization (W8A16, W4A8, etc.).
    
    Uses separate quantizers for weights and activations.
    
    Parameters
    ----------
    model : torch.nn.Module
        Original PyTorch model.
    weight_chop : Chop, Chopf, or Chopi
        Quantizer for weights.
    activation_chop : Chop, Chopf, or Chopi
        Quantizer for activations.
    calibration_data : iterable, optional
        Calibration data for static activation quantization.
    calibration_method : str, default='minmax'
        Calibration algorithm.
    dynamic : bool, default=True
        Use dynamic activation quantization (no calibration).
    eval_mode : bool, default=True
        Set model to eval mode.
    verbose : bool, default=False
        Print quantization details.
    
    Returns
    -------
    torch.nn.Module
        Quantized model with mixed-precision.
    
    Examples
    --------
    >>> from pychop import Chopi, Chop
    >>> from pychop.tch.ptq import mixed_post_quantization
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
    """
    from pychop.integer import Chopi
    
    # Step 1: Quantize weights
    q_model = post_quantization(model, weight_chop, eval_mode=eval_mode, verbose=verbose)
    
    # Step 2: Handle activation quantization
    if activation_chop is None:
        return q_model
    
    w_bits = getattr(weight_chop, 'bits', 'custom')
    a_bits = getattr(activation_chop, 'bits', 'custom')
    
    if verbose:
        print(f"[Mixed PTQ] Configuration: W{w_bits}A{a_bits}")
    
    if dynamic:
        # Dynamic activation quantization
        if verbose:
            print("[Mixed PTQ] Using dynamic activation quantization...")
        
        target_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear,
                         nn.ReLU, nn.GELU, nn.SiLU)
        
        for name, module in q_model.named_modules():
            if isinstance(module, target_layers):
                def act_hook(module, input, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    
                    if isinstance(activation_chop, Chopi):
                        q = activation_chop.quantize(output)
                        return activation_chop.dequantize(q)
                    elif hasattr(activation_chop, 'quantize'):
                        return activation_chop.quantize(output)
                    else:
                        return activation_chop(output)
                
                module.register_forward_hook(act_hook)
    
    else:
        # Static activation quantization (requires calibration)
        if calibration_data is None:
            raise ValueError("calibration_data is required for static activation quantization")
        
        if verbose:
            print(f"[Mixed PTQ] Using static activation quantization (method={calibration_method})...")
        
        # Collect statistics (same as static_post_quantization)
        stats = {}
        handles = []
        
        def make_calibration_hook(name):
            def hook(module, input, output):
                if not isinstance(output, torch.Tensor):
                    return
                
                output_data = output.detach()
                
                if name not in stats:
                    stats[name] = [output_data.min().item(), output_data.max().item()]
                else:
                    stats[name][0] = min(stats[name][0], output_data.min().item())
                    stats[name][1] = max(stats[name][1], output_data.max().item())
            
            return hook
        
        target_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear,
                         nn.ReLU, nn.GELU, nn.SiLU)
        
        for name, module in q_model.named_modules():
            if isinstance(module, target_layers):
                handle = module.register_forward_hook(make_calibration_hook(name))
                handles.append(handle)
        
        # Run calibration
        device = next(q_model.parameters()).device
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                inputs = inputs.to(device)
                q_model(inputs)
        
        # Remove calibration hooks
        for handle in handles:
            handle.remove()
        
        # Apply calibration algorithm
        if calibration_method == 'percentile':
            final_stats = calibrate_percentile({k: tuple(v) for k, v in stats.items()})
        else:
            final_stats = {k: tuple(v) for k, v in stats.items()}
        
        # Register static quantization hooks
        for name, module in q_model.named_modules():
            if name in final_stats:
                min_val, max_val = final_stats[name]
                
                def make_quant_hook(min_v, max_v):
                    def hook(module, input, output):
                        if not isinstance(output, torch.Tensor):
                            return output
                        
                        clamped = torch.clamp(output, min_v, max_v)
                        
                        if isinstance(activation_chop, Chopi):
                            q = activation_chop.quantize(clamped)
                            return activation_chop.dequantize(q)
                        elif hasattr(activation_chop, 'quantize'):
                            return activation_chop.quantize(clamped)
                        else:
                            return activation_chop(clamped)
                    
                    return hook
                
                module.register_forward_hook(make_quant_hook(min_val, max_val))
    
    if verbose:
        print(f"[Mixed PTQ] W{w_bits}A{a_bits} quantization complete!")
    
    return q_model