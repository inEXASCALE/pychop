"""Shared TensorFlow helpers for NumPy-backed custom-gradient wrappers."""

import tensorflow as tf


def ensure_tensor(x, dtype=None):
    """Convert a value to a TensorFlow tensor with an optional dtype cast.

    Parameters
    ----------
    x : array-like or tf.Tensor
        Value to convert.
    dtype : tf.DType, default=None
        Target dtype. If ``None``, the inferred dtype is preserved.

    Returns
    -------
    tf.Tensor
        Converted tensor.
    """
    tensor = tf.convert_to_tensor(x)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tf.cast(tensor, dtype)
    return tensor


def unary_numpy_op(x, numpy_fn, *, tout=None, identity_grad=True, shape_like=None):
    """Wrap a unary NumPy callback as a TensorFlow op.

    Parameters
    ----------
    x : array-like or tf.Tensor
        Input tensor.
    numpy_fn : callable
        NumPy function executed through ``tf.numpy_function``.
    tout : tf.DType, default=None
        TensorFlow dtype of the callback result.
    identity_grad : bool, default=True
        Whether to expose an identity straight-through gradient for floating
        inputs.
    shape_like : tf.Tensor, default=None
        Tensor whose static shape should be copied to the output.

    Returns
    -------
    tf.Tensor
        Callback result with optional identity gradient.
    """
    x = ensure_tensor(x)
    out_dtype = tout or x.dtype
    ref = shape_like if shape_like is not None else x

    if identity_grad and x.dtype.is_floating:
        @tf.custom_gradient
        def _op(inp):
            out = tf.numpy_function(lambda arr: numpy_fn(arr), [inp], Tout=out_dtype)
            out.set_shape(ref.shape)

            def grad(dy):
                return tf.cast(dy, inp.dtype)

            return out, grad

        return _op(x)

    out = tf.numpy_function(lambda arr: numpy_fn(arr), [x], Tout=out_dtype)
    out.set_shape(ref.shape)
    return out


def binary_numpy_op(x, y, numpy_fn, *, tout=None, grad_x=True, grad_y=False, shape_like=None):
    """Wrap a binary NumPy callback as a TensorFlow op.

    Parameters
    ----------
    x, y : array-like or tf.Tensor
        Input tensors passed to ``numpy_fn``.
    numpy_fn : callable
        NumPy function executed through ``tf.numpy_function``.
    tout : tf.DType, default=None
        TensorFlow dtype of the callback result.
    grad_x : bool, default=True
        Whether to pass an identity gradient through ``x``.
    grad_y : bool, default=False
        Whether to pass an identity gradient through ``y``.
    shape_like : tf.Tensor, default=None
        Tensor whose static shape should be copied to the output.

    Returns
    -------
    tf.Tensor
        Callback result with optional identity gradients.
    """
    x = ensure_tensor(x)
    y = ensure_tensor(y)
    out_dtype = tout or x.dtype
    ref = shape_like if shape_like is not None else x

    if (grad_x and x.dtype.is_floating) or (grad_y and y.dtype.is_floating):
        @tf.custom_gradient
        def _op(a, b):
            out = tf.numpy_function(lambda arr_a, arr_b: numpy_fn(arr_a, arr_b), [a, b], Tout=out_dtype)
            out.set_shape(ref.shape)

            def grad(dy):
                gx = tf.cast(dy, a.dtype) if grad_x else None
                gy = tf.cast(dy, b.dtype) if grad_y else None
                return gx, gy

            return out, grad

        return _op(x, y)

    out = tf.numpy_function(lambda arr_a, arr_b: numpy_fn(arr_a, arr_b), [x, y], Tout=out_dtype)
    out.set_shape(ref.shape)
    return out
