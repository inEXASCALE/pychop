"""Native tensor parity, graph compilation, validation and STE contracts."""
import importlib
import os
import numpy as np
import pytest
from pychop import P3109, P3109Format


@pytest.fixture(params=["torch", "tensorflow", "jax"])
def framework(request):
    lib = pytest.importorskip(request.param)
    if request.param == "jax":
        # Context-local test setting; the library never modifies JAX configuration.
        with lib.enable_x64(True):
            yield request.param, lib
    else:
        yield request.param, lib


def tensor(framework, x, dtype="float64"):
    name, lib = framework
    if name == "torch":
        return lib.tensor(x, dtype=getattr(lib, dtype))
    if name == "tensorflow":
        return lib.constant(x, dtype=getattr(lib, dtype))
    return lib.numpy.asarray(x, dtype=getattr(lib.numpy, dtype))


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@pytest.mark.parametrize("p,signed,domain", [(1, True, "extended"), (4, True, "finite"),
                                            (4, True, "extended"), (4, False, "extended"),
                                            (8, False, "finite")])
def test_codepoints_and_midpoint_parity(framework, p, signed, domain):
    fmt = P3109Format(8, p, signed, domain)
    q = P3109(fmt)
    code = np.arange(256)
    decoded = q.decode(code)
    np.testing.assert_array_equal(array(q.decode(tensor(framework, code, "int64"))), decoded)
    np.testing.assert_array_equal(array(q.encode(tensor(framework, decoded))), code)
    finite = np.sort(decoded[np.isfinite(decoded)])
    mid = finite[:-1] + (finite[1:] - finite[:-1]) / 2
    x = np.r_[mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf),
              [0, -0., -1e-300, 1e-300, fmt.max * 2, -fmt.max * 2, np.inf, -np.inf, np.nan]]
    for mode in ["nearest_even", "nearest_away", "toward_zero", "toward_positive", "toward_negative", "to_odd"]:
        for sat in [False, True, "propagate"]:
            policy = P3109(fmt, mode, sat)
            actual = array(policy(tensor(framework, x)))
            np.testing.assert_array_equal(actual, policy(x), err_msg=f"{framework[0]} {mode} {sat}")
            assert not np.any(np.signbit(actual[actual == 0]))


@pytest.mark.parametrize("mode", ["stochastic_a", "stochastic_b", "stochastic_c"])
def test_stochastic_parity(framework, mode):
    rng = np.random.default_rng(824)
    x = rng.normal(size=1024)
    bits = rng.integers(0, 256, size=x.shape)
    q = P3109(rounding=mode)
    actual = q(tensor(framework, x), srbits=tensor(framework, bits, "int64"), srnumbits=8)
    np.testing.assert_array_equal(array(actual), q(x, srbits=bits, srnumbits=8))
    # Compile call sites may opt out of runtime assertions; invalid elements remain NaN.
    invalid = q(tensor(framework, [1.1]), srbits=tensor(framework, [256], "int64"), srnumbits=8, check=False)
    assert np.isnan(array(invalid)).all()


def test_float32_half_strides_and_subnormal_input(framework):
    x = np.array([1.0625, -1.1, np.nextafter(np.float32(0), np.float32(1)),
                  -np.nextafter(np.float32(0), np.float32(1)), np.inf, np.nan], dtype=np.float32)
    for mode in ["nearest_even", "toward_positive", "toward_negative", "to_odd"]:
        q = P3109(rounding=mode)
        actual = q(tensor(framework, x, "float32"))
        assert "float32" in str(actual.dtype)
        np.testing.assert_array_equal(array(actual), q(x))
    assert "float32" in str(P3109()(tensor(framework, [1.1], "float16")).dtype)
    a = tensor(framework, np.arange(48).reshape(6, 8), "float32")[:, ::2]
    np.testing.assert_array_equal(array(P3109()(a)), P3109()(array(a)))
    for shape in [(), (0,), (2, 0)]:
        x = np.zeros(shape, dtype=np.float32)
        assert tuple(P3109()(tensor(framework, x, "float32")).shape) == shape


def test_validation(framework):
    name, lib = framework
    errors = (ValueError, TypeError) + ((lib.errors.InvalidArgumentError,) if name == "tensorflow" else ())
    for codes in [[-1], [256]]:
        with pytest.raises(errors):
            P3109().decode(tensor(framework, codes, "int64"))
        assert np.isnan(array(P3109().decode(tensor(framework, codes, "int64"), check=False))).all()
    with pytest.raises(TypeError):
        P3109().decode(tensor(framework, [1.2]))
    with pytest.raises(ValueError, match="float64"):
        P3109(P3109Format(8, 1, False))(tensor(framework, [1.], "float32"))
    with pytest.raises(errors):
        P3109(rounding="stochastic_a")(tensor(framework, [1.]), srbits=tensor(framework, [-1], "int64"), srnumbits=8)


def test_graph_and_ste(framework):
    name, lib = framework
    q = P3109(saturate=True)
    data = np.array([1.0625, -.1, 300., np.inf, np.nan], dtype=np.float32)
    x = tensor(framework, data, "float32")
    if name == "torch":
        x.requires_grad_()
        y = q(x, ste=True)
        y.sum().backward()
        np.testing.assert_array_equal(array(x.grad), np.ones_like(data))
        assert not q(x).requires_grad
        compiled = lib.compile(lambda a: q(a, ste=True), backend=os.environ.get("PYCHOP_TORCH_COMPILE_BACKEND", "eager"), fullgraph=True)
        np.testing.assert_array_equal(array(compiled(x)), q(data))
        assert q(x).device == x.device
    elif name == "tensorflow":
        with lib.GradientTape() as tape:
            tape.watch(x)
            loss = lib.reduce_sum(q(x, ste=True))
        np.testing.assert_array_equal(array(tape.gradient(loss, x)), np.ones_like(data))
        compiled = lib.function(lambda a: q(a, ste=True), autograph=False)
        np.testing.assert_array_equal(array(compiled(x)), q(data))
        with lib.GradientTape() as tape:
            tape.watch(x)
            loss = lib.reduce_sum(q(x))
        assert tape.gradient(loss, x) is None
    else:
        grad = lib.grad(lambda a: lib.numpy.sum(q(a, ste=True)))(x)
        np.testing.assert_array_equal(array(grad), np.ones_like(data))
        np.testing.assert_array_equal(array(lib.grad(lambda a: q(a).sum())(x)), np.zeros_like(data))
        compiled = lib.jit(lambda a: q(a, ste=True))
        np.testing.assert_array_equal(array(compiled(x)), q(data))
        np.testing.assert_array_equal(array(lib.vmap(q)(lib.numpy.stack([x, x]))), q(np.stack([data, data])))
    encode = lambda a: q.encode(a, check=False)
    decode = lambda a: q.decode(a, check=False)
    if name == "torch":
        encoded = lib.compile(encode, backend=os.environ.get("PYCHOP_TORCH_COMPILE_BACKEND", "eager"), fullgraph=True)(x)
        restored = lib.compile(decode, backend=os.environ.get("PYCHOP_TORCH_COMPILE_BACKEND", "eager"), fullgraph=True)(encoded)
    elif name == "tensorflow":
        encoded = lib.function(encode, autograph=False)(x)
        restored = lib.function(decode, autograph=False)(encoded)
    else:
        encoded = lib.jit(encode)(x)
        restored = lib.jit(decode)(encoded)
    np.testing.assert_array_equal(array(restored), q(data))


def test_jax_default_float32_mode():
    jax = pytest.importorskip("jax")
    with jax.enable_x64(False):
        q = P3109()
        x = jax.numpy.array([1.0625, -.1], dtype=jax.numpy.float32)
        np.testing.assert_array_equal(array(jax.jit(q)(x)), q(np.asarray(x)))
        decoded = jax.jit(lambda a: q.decode(q.encode(a), check=False))(x)
        np.testing.assert_array_equal(array(decoded), q(np.asarray(x)))


@pytest.mark.parametrize("dtype,n", [("float32", 23), ("float64", 32)])
def test_compiled_stochastic_and_boundary_bits(framework, dtype, n):
    name, lib = framework
    # Exercise every random-bit rounding boundary around a target midpoint.
    x = np.array([0., 1., 1.0625, -1.0625, 1.125, np.inf, np.nan])
    b = np.array([0, 2**n - 1, 2**(n-1)-1, 2**(n-1), 1, 0, 0], dtype=np.int64)
    a, bits = tensor(framework, x, dtype), tensor(framework, b, "int64")
    for mode in ["stochastic_a", "stochastic_b", "stochastic_c"]:
        q = P3109(rounding=mode)
        fn = lambda v, r: q(v, srbits=r, srnumbits=n, check=False)
        if name == "torch":
            compiled = lib.compile(fn, backend=os.environ.get("PYCHOP_TORCH_COMPILE_BACKEND", "eager"), fullgraph=True)
        elif name == "tensorflow":
            compiled = lib.function(fn, autograph=False)
        else:
            compiled = lib.jit(fn)
        np.testing.assert_array_equal(array(compiled(a, bits)), q(x, srbits=b, srnumbits=n))


def test_wide_formats_and_decode_dtype(framework):
    rng = np.random.default_rng(26)
    for fmt in [P3109Format(16, 11), P3109Format(32, 23), P3109Format(32, 32, False)]:
        q = P3109(fmt)
        code = rng.integers(0, 2**fmt.k, size=500, dtype=np.int64)
        native_code = tensor(framework, code, "int64")
        decoded = q.decode(native_code)
        np.testing.assert_array_equal(array(decoded), q.decode(code))
        np.testing.assert_array_equal(array(q.encode(decoded)), code)
    q = P3109()
    codes = tensor(framework, [0, 64, 128], "int64")
    assert "float32" in str(q.decode(codes, dtype="float32").dtype)
    with pytest.raises(ValueError):
        q.decode(codes, dtype="float16")
    with pytest.raises((TypeError, ValueError, RuntimeError) + ((framework[1].errors.InvalidArgumentError,) if framework[0] == "tensorflow" else ())):
        P3109(rounding="stochastic_c")(tensor(framework, [1., 2.]), srbits=tensor(framework, [[1], [2]], "int64"), srnumbits=8)


def test_torch_optional_accelerator():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")
    x = torch.tensor([1.0625, -.1, float('inf')], device="cuda", requires_grad=True)
    q = P3109()
    result = q(x, ste=True)
    assert result.device == x.device
    np.testing.assert_array_equal(array(result), q(array(x)))
    np.testing.assert_array_equal(array(q.decode(q.encode(x), dtype="float32")), q(array(x)))


def test_training_export(framework, tmp_path):
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples" / "p3109" / "tensor_training.py"
    spec = importlib.util.spec_from_file_location("p3109_tensor_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.run(framework[0], tmp_path)
    assert metrics["final_training_loss"] < .1 * metrics["initial_loss"]
    assert metrics["exported_model_mse"] < .01



def test_uint8_random_bits_and_codes(framework):
    q = P3109(rounding="stochastic_c")
    x = np.full(3, 1.0625)
    bits = np.array([0, 128, 255], dtype=np.uint8)
    actual = q(tensor(framework, x), srbits=tensor(framework, bits, "uint8"), srnumbits=8)
    np.testing.assert_array_equal(array(actual), q(x, srbits=bits, srnumbits=8))
    codes = tensor(framework, [0, 128, 255], "uint8")
    np.testing.assert_array_equal(array(P3109().decode(codes)), P3109().decode(np.array([0, 128, 255])))
