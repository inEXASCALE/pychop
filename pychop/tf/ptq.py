import numpy as np
import tensorflow as tf


def _as_tensor(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    return tf.convert_to_tensor(x)


def _iter_calibration_inputs(calibration_data):
    for batch in calibration_data:
        yield _as_tensor(batch)


def _compute_bounds_from_samples(samples, calibration_method='minmax', percentile=99.99):
    if not samples:
        return None

    valid = [np.ravel(s) for s in samples if s.size > 0]
    if not valid:
        return None

    arr = np.concatenate(valid, axis=0)
    if arr.size == 0:
        return None

    method = calibration_method.lower()
    if method == 'minmax':
        return float(np.min(arr)), float(np.max(arr))
    if method == 'percentile':
        low = float(np.percentile(arr, 100.0 - percentile))
        high = float(np.percentile(arr, percentile))
        return low, high
    if method == 'mse':
        max_abs = float(np.max(np.abs(arr)))
        candidates = np.linspace(0.5 * max_abs, max_abs, 25)
        best = max_abs
        best_mse = np.inf
        for t in candidates:
            clipped = np.clip(arr, -t, t)
            mse = float(np.mean((arr - clipped) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = t
        return -best, best
    if method == 'kl_divergence':
        # Lightweight approximation: use a high percentile symmetric threshold.
        t = float(np.percentile(np.abs(arr), percentile))
        return -t, t
    raise ValueError(
        f"Unsupported calibration_method '{calibration_method}'. "
        "Expected one of: minmax, percentile, kl_divergence, mse."
    )


class _PTQWrapper(tf.keras.Model):
    def __init__(self, base_model, activation_chop=None, activation_clip=None,
                 layer_clips=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.activation_chop = activation_chop
        self.activation_clip = activation_clip
        self.layer_clips = layer_clips or {}

    def call(self, inputs, training=False, **kwargs):
        if self.layer_clips or (self.activation_chop is not None and not self.activation_clip):
            # Per-layer quantization is handled by _PerLayerQuantWrapper sub-layers
            pass
        outputs = self.base_model(inputs, training=training, **kwargs)
        if self.activation_clip is not None:
            lo, hi = self.activation_clip
            outputs = tf.clip_by_value(outputs, lo, hi)
        if self.activation_chop is not None:
            outputs = self.activation_chop(outputs)
        return outputs


class _PerLayerQuantModel(tf.keras.Model):
    """Wraps a model and applies activation quantization after each target layer."""

    _TARGET_LAYER_TYPES = (
        tf.keras.layers.Dense,
        tf.keras.layers.Conv1D, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.LayerNormalization,
    )

    def __init__(self, base_model, chop, layer_clips=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.chop = chop
        self.layer_clips = layer_clips or {}
        self._target_layer_names = set()
        for layer in base_model.layers:
            if isinstance(layer, self._TARGET_LAYER_TYPES):
                self._target_layer_names.add(layer.name)

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for layer in self.base_model.layers:
            try:
                x = layer(x, training=training) if _layer_accepts_training(layer) else layer(x)
            except TypeError:
                x = layer(x)
            if layer.name in self._target_layer_names and self.chop is not None:
                if layer.name in self.layer_clips:
                    lo, hi = self.layer_clips[layer.name]
                    x = tf.clip_by_value(x, lo, hi)
                x = self.chop(x)
        return x


def _layer_accepts_training(layer):
    """Check if a layer's call method accepts a 'training' argument."""
    import inspect
    sig = inspect.signature(layer.call)
    return 'training' in sig.parameters


def _clone_with_quantized_weights(model, weight_chop=None):
    cloned = tf.keras.models.clone_model(model)
    if hasattr(model, 'input_shape') and model.input_shape is not None:
        try:
            cloned.build(model.input_shape)
        except Exception:
            pass

    weights = model.get_weights()
    if not weights:
        return cloned

    if weight_chop is None:
        cloned.set_weights(weights)
        return cloned

    quantized = []
    for weight in weights:
        tensor = tf.convert_to_tensor(weight)
        q = weight_chop(tensor)
        quantized.append(tf.convert_to_tensor(q).numpy())
    cloned.set_weights(quantized)
    return cloned


def _collect_activation_samples(model, calibration_data):
    samples = []
    for x in _iter_calibration_inputs(calibration_data):
        y = model(x, training=False)
        samples.append(tf.reshape(tf.cast(y, tf.float32), [-1]).numpy())
    return samples


def post_quantization(model, chop, eval_mode=True, verbose=False):
    del eval_mode, verbose
    return _clone_with_quantized_weights(model, weight_chop=chop)


def dynamic_post_quantization(model, chop, eval_mode=True, verbose=False):
    del eval_mode
    quantized = _clone_with_quantized_weights(model, weight_chop=chop)
    if verbose:
        print("[TF Dynamic PTQ] Weights quantized. Activation quantization applied per-layer.")
    return _PerLayerQuantModel(quantized, chop=chop)


def _collect_per_layer_activation_samples(model, calibration_data):
    """Collect activation samples per target layer for calibration."""
    target_types = _PerLayerQuantModel._TARGET_LAYER_TYPES
    layer_samples = {}

    for x in _iter_calibration_inputs(calibration_data):
        intermediate = x
        for layer in model.layers:
            try:
                intermediate = layer(intermediate, training=False) if _layer_accepts_training(layer) else layer(intermediate)
            except TypeError:
                intermediate = layer(intermediate)
            if isinstance(layer, target_types):
                out_np = tf.reshape(tf.cast(intermediate, tf.float32), [-1]).numpy()
                if layer.name not in layer_samples:
                    layer_samples[layer.name] = []
                layer_samples[layer.name].append(out_np)
    return layer_samples


def static_post_quantization(model, chop, calibration_data, calibration_method='minmax', percentile=99.99,
                             fuse_bn=True, eval_mode=True, verbose=False, model_apply_fn=None):
    del fuse_bn, eval_mode, model_apply_fn
    quantized = _clone_with_quantized_weights(model, weight_chop=chop)

    layer_clips = {}
    if calibration_data:
        layer_samples = _collect_per_layer_activation_samples(quantized, calibration_data)
        for layer_name, samples in layer_samples.items():
            bounds = _compute_bounds_from_samples(samples, calibration_method=calibration_method, percentile=percentile)
            if bounds is not None:
                layer_clips[layer_name] = bounds
                if verbose:
                    print(f"[TF Static PTQ] {layer_name} clip range: [{bounds[0]:.6f}, {bounds[1]:.6f}]")

    # Also collect overall output-level stats for backward compatibility
    samples = _collect_activation_samples(quantized, calibration_data)
    clip_bounds = _compute_bounds_from_samples(samples, calibration_method=calibration_method, percentile=percentile)
    if verbose and clip_bounds is not None:
        print(f"TensorFlow static PTQ activation clip range: [{clip_bounds[0]:.6f}, {clip_bounds[1]:.6f}]")

    return _PerLayerQuantModel(quantized, chop=chop, layer_clips=layer_clips)


def mixed_post_quantization(model, weight_chop, activation_chop, calibration_data=None,
                            calibration_method='minmax', percentile=99.99,
                            dynamic=True, eval_mode=True, verbose=False):
    del eval_mode
    quantized = _clone_with_quantized_weights(model, weight_chop=weight_chop)

    layer_clips = {}
    if not dynamic and activation_chop is not None:
        if calibration_data is None:
            raise ValueError("calibration_data is required when dynamic=False for TensorFlow mixed PTQ.")
        layer_samples = _collect_per_layer_activation_samples(quantized, calibration_data)
        for layer_name, samples in layer_samples.items():
            bounds = _compute_bounds_from_samples(samples, calibration_method=calibration_method, percentile=percentile)
            if bounds is not None:
                layer_clips[layer_name] = bounds
                if verbose:
                    print(f"[TF Mixed PTQ] {layer_name} clip range: [{bounds[0]:.6f}, {bounds[1]:.6f}]")

    if activation_chop is not None:
        return _PerLayerQuantModel(quantized, chop=activation_chop, layer_clips=layer_clips)

    return _PTQWrapper(quantized)
