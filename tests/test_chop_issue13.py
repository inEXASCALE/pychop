"""Issue #13: IEEE-style conversion boundaries for Chop and LightChopSTE."""

import warnings

import numpy as np
import pytest

from pychop import Chop


BACKENDS = ("numpy", "torch", "jax", "tensorflow")


def _convert(backend, values, monkeypatch, *, exp_bits=5, sig_bits=10,
             rmode=1, dtype=np.float64, subnormal=True):
    array = np.asarray(values, dtype=dtype)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        array = torch.as_tensor(array.copy())
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        array = jnp.asarray(array)
    elif backend == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        array = tf.convert_to_tensor(array)
    monkeypatch.setenv("chop_backend", backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = Chop(exp_bits, sig_bits, rmode=rmode, subnormal=subnormal)(array)
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)


ISSUE_13_CASES = (
    pytest.param(70000.0, np.inf, 1, 5, 10, id="large-finite-to-infinity"),
    pytest.param(100000.0, np.inf, 1, 5, 10, id="all-ones-input-exponent"),
    pytest.param(1e6, np.inf, 1, 5, 10, id="far-over-range"),
    pytest.param(65520.0, np.inf, 1, 5, 10, id="rounding-carry-to-infinity"),
    pytest.param(np.finfo(np.float64).max, 65504.0, 4, 5, 10, id="toward-zero-overflow-saturates"),
    pytest.param(np.nextafter(8.0, 0.0), 7.99609375, 4, 5, 10, id="just-below-power-of-two"),
    pytest.param(np.nextafter(3 * 2**-25, 0.0), 2**-24, 1, 5, 10, id="just-below-subnormal-midpoint"),
    pytest.param(5e-324, 2**-24, 2, 5, 10, id="tiny-positive-directed-up"),
    pytest.param(1e-300, 1e-300, 1, 11, 52, id="binary64-small-normal-exact"),
    pytest.param(4.6663180925160944e-302, 4.6663180925160944e-302, 1, 11, 52,
                 id="reported-binary64-small-normal"),
    pytest.param(5e-324, 5e-324, 1, 11, 52, id="binary64-source-subnormal-exact"),
    pytest.param(-2.0009765625, -2.001953125, 8, 5, 10,
                 id="reported-negative-tie-away"),
    pytest.param(2.0009765624999996, 2.0, 8, 5, 10,
                 id="reported-near-tie-below"),
    pytest.param(-(1 + 2**-11), -(1 + 2**-10), 8, 5, 10, id="negative-tie-away"),
    pytest.param(1 + 2**-11 - 2**-40, 1.0, 8, 5, 10, id="near-tie-below"),
    pytest.param(-(1 + 2**-11), -1.0, 7, 5, 10, id="negative-tie-to-zero"),
    pytest.param(1 + 2**-11 + 2**-40, 1 + 2**-10, 7, 5, 10, id="near-tie-above"),
    pytest.param(1.0, 1.0, 9, 5, 10, id="round-to-odd-exact-one"),
    pytest.param(2.0, 2.0, 9, 5, 10, id="round-to-odd-exact-two"),
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value,expected,rmode,exp_bits,sig_bits", ISSUE_13_CASES)
def test_issue_13_reported_conversions(
    backend, value, expected, rmode, exp_bits, sig_bits, monkeypatch
):
    result = _convert(backend, [value], monkeypatch, exp_bits=exp_bits,
                      sig_bits=sig_bits, rmode=rmode)
    np.testing.assert_array_equal(result, [expected],
                                  err_msg=f"{backend}, rmode={rmode}, input={value!r}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode,expected", (
    (1, [np.inf, -np.inf]),
    (2, [np.inf, -65504.0]),
    (3, [65504.0, -np.inf]),
    (4, [65504.0, -65504.0]),
))
def test_overflow_direction_and_signed_zero(backend, rmode, expected, monkeypatch):
    result = _convert(backend, [70000.0, -70000.0, -0.0, 0.0],
                      monkeypatch, rmode=rmode)
    np.testing.assert_array_equal(result[:2], expected)
    np.testing.assert_array_equal(np.signbit(result[2:]), [True, False])


@pytest.mark.parametrize("backend", BACKENDS)
def test_binary16_nearest_matches_numpy(backend, monkeypatch):
    rng = np.random.default_rng(13)
    values = np.concatenate((
        rng.uniform(-70000, 70000, 128),
        rng.uniform(-1e-4, 1e-4, 128),
        [0.0, -0.0, 65504.0, 65519.0, 65520.0, 2**-25, 2**-24,
         np.inf, -np.inf, np.nan],
    ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = values.astype(np.float16).astype(np.float64)
    actual = _convert(backend, values, monkeypatch)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_max_and_source_subnormal(backend, monkeypatch):
    maximum = np.finfo(np.float32).max
    result = _convert(backend, [maximum, np.nextafter(np.float32(0), np.float32(1))],
                      monkeypatch, exp_bits=8, sig_bits=23, dtype=np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [maximum, np.nextafter(np.float32(0), np.float32(1))])


@pytest.mark.parametrize("backend", BACKENDS)
def test_multidimensional_and_no_subnormal_mode(backend, monkeypatch):
    values = np.array([[0.0, -0.0, 2**-25], [2**-24, 2**-14, 1.0]])
    result = _convert(backend, values, monkeypatch, subnormal=False)
    np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0], [0.0, 2**-14, 1.0]])
    np.testing.assert_array_equal(np.signbit(result[0, :2]), [False, True])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rmode,expected", (
    (7, [1.0, 1 + 2**-10]),
    (8, [1 + 2**-10, 1 + 2**-10]),
    (9, [1 + 2**-10, 1 + 2**-10]),
))
def test_special_rounding_modes_at_tie_and_inexact(backend, rmode, expected, monkeypatch):
    values = [1 + 2**-11, 1 + 2**-11 + 2**-20]
    result = _convert(backend, values, monkeypatch, rmode=rmode)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_private_component_entry_points_use_exact_exponents(backend):
    values = np.array([5e-324, 1e-300, np.nextafter(8.0, 0.0)])
    if backend == "numpy":
        from pychop.np.lightchop import LightChop_
        x = values
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        from pychop.tch.lightchop import LightChop_
        x = torch.as_tensor(values)
    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from pychop.jx.lightchop import LightChop_
        x = jnp.asarray(values)
    converter = LightChop_(5, 10, rmode=2)
    components = converter._to_custom_float(x)
    np.testing.assert_array_equal(np.asarray(components[1])[:2], [0, 0])
    assert np.all(np.asarray(components[2])[:2] > 0)
    if backend == "jax":
        result = converter._quantize_impl(x, *components, 2, None)
    elif backend == "numpy":
        result = converter._quantize_components(x, *components, 2)
    else:
        result = converter._quantize_components(x, *components)
    np.testing.assert_array_equal(np.asarray(result), [2**-24, 2**-24, 8.0])


def test_torch_ste_uses_correct_forward_and_identity_gradient():
    torch = pytest.importorskip("torch")
    from pychop.tch.lightchop import LightChopSTE

    source = torch.tensor([70000.0, 65520.0, 2**-24, -0.0, 1.0],
                          dtype=torch.float64, requires_grad=True)
    result = LightChopSTE(5, 10)(source)
    np.testing.assert_array_equal(result.detach().numpy()[:3], [np.inf, np.inf, 2**-24])
    assert bool(torch.signbit(result[3]))
    result.sum().backward()
    torch.testing.assert_close(source.grad[:3], torch.ones(3, dtype=torch.float64))


def test_jax_chop_keeps_callers_x64_setting(monkeypatch):
    jax = pytest.importorskip("jax")
    from pychop._jax_compat import enable_x64
    import jax.numpy as jnp

    monkeypatch.setenv("chop_backend", "jax")
    with enable_x64(False):
        source = jnp.array([1.5], dtype=jnp.float32)
        for exp_bits, sig_bits in ((5, 10), (11, 52)):
            result = Chop(exp_bits, sig_bits)(source)
            np.testing.assert_array_equal(np.asarray(result), [1.5])
            assert not jax.config.jax_enable_x64


@pytest.mark.parametrize("exp_bits,sig_bits", ((5, 10), (11, 10), (11, 52)))
def test_jax_chop_jit_matches_eager_without_global_x64(
    exp_bits, sig_bits, monkeypatch
):
    jax = pytest.importorskip("jax")
    from pychop._jax_compat import enable_x64
    import jax.numpy as jnp

    monkeypatch.setenv("chop_backend", "jax")
    with enable_x64(False):
        source = jnp.asarray([1.1, 70000.0, 2**-149, -0.0], dtype=jnp.float32)
        converter = Chop(exp_bits, sig_bits)
        expected = np.asarray(converter(source))
        with jax.checking_leaks():
            actual = np.asarray(jax.jit(converter)(source))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        assert not jax.config.jax_enable_x64


def test_jax_chop_jit_float64_subnormal(monkeypatch):
    jax = pytest.importorskip("jax")
    from pychop._jax_compat import enable_x64
    import jax.numpy as jnp

    monkeypatch.setenv("chop_backend", "jax")
    with enable_x64(True):
        source = jnp.asarray([5e-324, 1e-300, np.nextafter(8.0, 0.0)])
        converter = Chop(5, 10, rmode=2)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(converter)(source)),
            np.asarray(converter(source)),
        )
