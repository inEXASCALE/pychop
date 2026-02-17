# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np

import jax
import jax.numpy as jnp

import ml_dtypes

import gfloat
from gfloat.formats import *


def test_jax() -> None:
    """
    Test that JAX JIT produces correct output
    """
    a = np.random.randn(1024)

    a8 = a.astype(ml_dtypes.float8_e5m2).astype(jnp.float32)

    fi = format_info_ocp_e5m2
    j8 = gfloat.round_ndarray(fi, jnp.array(a)).astype(jnp.float32)  # type: ignore [arg-type]

    np.testing.assert_equal(a8, j8)

    jax_round_array = jax.jit(lambda x: gfloat.round_ndarray(fi, x))
    j8i = jax_round_array(a).astype(jnp.float32)

    np.testing.assert_equal(a8, j8i)
