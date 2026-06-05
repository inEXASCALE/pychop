import tensorflow as tf


def ensure_tensor(x, dtype=None):
    tensor = tf.convert_to_tensor(x)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tf.cast(tensor, dtype)
    return tensor


def unary_numpy_op(x, numpy_fn, *, tout=None, identity_grad=True, shape_like=None):
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
