# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Block floating point formats
# https://en.wikipedia.org/wiki/Block_floating_point

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

import array_api_compat

from .decode import decode_float
from .round import RoundMode, round_float
from .encode import encode_float
from .types import FormatInfo


@dataclass
class BlockFormatInfo:

    #: Short name for the format, e.g. BlockFP8
    name: str

    #: Element data type
    etype: FormatInfo

    #: Scaling block size
    k: int

    #: Scale datatype
    stype: FormatInfo

    #: ## Derived values

    @property
    def element_bits(self) -> int:
        """The number of bits in each element, d"""
        return self.etype.k

    @property
    def scale_bits(self) -> int:
        """The number of bits in the scale, w"""
        return self.stype.k

    @property
    def block_size_bytes(self) -> int:
        """The number of bytes in a block"""
        bits = self.element_bits * self.k + self.scale_bits
        assert bits % 8 == 0
        return bits // 8

    @property
    def __name__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"BlockFormatInfo:{self.name})"


def decode_block(fi: BlockFormatInfo, block: Iterable[int]) -> Iterable[float]:
    """
    Decode a :paramref:`block` of integer codepoints in Block Format :paramref:`fi`

    The scale is encoded in the first value of :paramref:`block`,
    with the remaining values encoding the block elements.

    The size of the iterable is not checked against the format descriptor.

    Args:
      fi (BlockFormatInfo): Describes the block format
      block (Iterable[int]): Input block

    Returns:
      A sequence of floats representing the encoded values.
    """
    it = iter(block)

    scale_encoding = next(it)
    scale = decode_float(fi.stype, scale_encoding).fval

    for val_encoding in it:
        val = scale * decode_float(fi.etype, val_encoding).fval
        yield val

    # TODO: Assert length of block was k+1?  Messy unless block is len()able


def encode_block(
    fi: BlockFormatInfo,
    scale: float,
    vals: Iterable[float],
    round: RoundMode = RoundMode.TiesToEven,
) -> Iterable[int]:
    """
    Encode float :paramref:`vals` into block Format described by :paramref:`fi`

    The :paramref:`scale` is explicitly passed, and the :paramref:`vals` are
    assumed to already be multiplied by `1/scale`.
    That is, this is pure encoding, scaling is computed and applied elsewhere
    (see e.g. :func:`quantize_block`).

    It is checked for overflow in the target format,
    and will raise an exception if it does.

    Args:
      fi (BlockFormatInfo): Describes the target block format
      scale (float): Scale to be recorded in the block
      vals (Iterable[float]): Input block
      round (RoundMode): Rounding mode to use, defaults to `TiesToEven`

    Returns:
      A sequence of ints representing the encoded values.

    Raises:
      ValueError: The scale overflows the target scale encoding format.
    """

    if scale > fi.stype.max or scale < fi.stype.min:
        raise ValueError(f"Scaled {scale} out of range for {fi.stype}")

    sat = True  # Saturate elements if out of range

    def enc(ty: FormatInfo, x: float) -> int:
        return encode_float(ty, round_float(ty, x, round, sat))

    yield enc(fi.stype, scale)

    for val in vals:
        yield enc(fi.etype, val)


ComputeScaleCallable = Callable[[float, npt.ArrayLike], float]


def compute_scale_amax(emax: float, vals: npt.ArrayLike) -> float:
    """
    Compute a scale factor such that :paramref:`vals` can be scaled to the
    range [0, 2**emax].  That is, `scale` is computed such that the largest
    exponent in the array `vals * scale` will be `emax`.

    The scale is clipped to the range 2**[-127, 127].

    If all values are zero, any scale value smaller than emax would be accurate,
    but returning the smallest possible means that quick checks on the magnitude
    to identify near-zero blocks will also find the all-zero blocks.

    Args:
      emax (float): Maximum exponent to appear in `vals * scale`
      vals (ArrayLike): Input block

    Returns:
      A float such that `vals * scale` has exponents less than or equal to `emax`.

    Note:
      If all vals are zero, 1.0 is returned.
    """
    xp = array_api_compat.array_namespace(vals)

    amax = xp.max(xp.abs(vals))
    if amax == 0.0:
        q_log2scale = -127.0
    else:
        q_log2scale = xp.floor(xp.log2(amax)) - emax
        q_log2scale = xp.clip(q_log2scale, -127.0, 127.0)
    return 2.0**q_log2scale


def quantize_block(
    fi: BlockFormatInfo,
    vals: npt.NDArray,
    compute_scale: ComputeScaleCallable,
    round: RoundMode = RoundMode.TiesToEven,
) -> npt.NDArray:
    """
    Encode and decode a block of :paramref:`vals` of bytes into
    block format described by :paramref:`fi`

    Args:
      fi (BlockFormatInfo): Describes the target block format
      vals (numpy.array): Input block
      compute_scale ((float, ArrayLike) -> float):
          Callable to compute the scale, defaults to :func:`compute_scale_amax`
      round (RoundMode): Rounding mode to use, defaults to `TiesToEven`

    Returns:
      An array of floats representing the quantized values.

    Raises:
      ValueError: The scale overflows the target scale encoding format.
    """

    q_scale = compute_scale(fi.etype.emax, vals)
    scaled_vals = vals / q_scale
    enc = encode_block(fi, q_scale, scaled_vals, round)
    return np.fromiter(decode_block(fi, enc), float)
