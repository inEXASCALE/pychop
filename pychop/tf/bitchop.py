"""TensorFlow bit-level floating-point chopping backend."""

import tensorflow as tf

from ..np.bitchop import Bitchop as NPBitchop
from .common import unary_numpy_op


class Bitchop:
    """TensorFlow adapter for NumPy bit-level floating-point chopping.

    Parameters
    ----------
    exp_bits : int
        Number of exponent bits in the simulated format.
    sig_bits : int
        Number of significand precision bits.
    rmode : str, default='nearest_even'
        Rounding mode forwarded to the NumPy ``Bitchop`` implementation.
    subnormal : bool, default=True
        Whether subnormal values are represented.
    random_state : int, default=42
        Seed used by stochastic rounding modes.
    device : str, default='cpu'
        Kept for API compatibility.
    """

    def __init__(self, exp_bits, sig_bits, rmode='nearest_even', subnormal=True, random_state=42, device='cpu'):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.subnormal = subnormal
        self.random_state = random_state
        self.device = device
        self._np_impl = NPBitchop(
            exp_bits=exp_bits,
            sig_bits=sig_bits,
            rmode=rmode,
            subnormal=subnormal,
            random_state=random_state,
        )

    def __call__(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)
        return unary_numpy_op(
            x,
            lambda arr: self._np_impl(arr),
            tout=x.dtype,
            identity_grad=x.dtype.is_floating,
        )
