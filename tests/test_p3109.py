import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pychop import P3109, P3109Format, p3109_decode, p3109_encode, p3109_quantize

FORMATS = [(k, p, signed, domain) for k in range(3, 9) for signed in (True, False)
           for p in range(1, k + (not signed)) for domain in ("finite", "extended")]
MODES = ["nearest_even", "nearest_away", "toward_zero", "toward_positive", "toward_negative"]


@pytest.mark.parametrize("args", FORMATS)
def test_all_codepoints_roundtrip(args):
    fmt = P3109Format(*args)
    codes = np.arange(1 << fmt.k, dtype=np.uint32)
    values = p3109_decode(codes, fmt)
    np.testing.assert_array_equal(p3109_encode(values, fmt), codes)
    assert np.count_nonzero(np.isnan(values)) == 1
    assert not np.signbit(values[0])


@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("domain", ["finite", "extended"])
def test_public_value_tables(signed, domain):
    fmt = P3109Format(8, 4, signed, domain)
    filename = f"Binary8p4{'s' if signed else 'u'}{domain[0]}.csv"
    with (Path(__file__).parent / "data" / "p3109" / filename).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = [float.fromhex(r["value"]) for r in rows]
    actual = p3109_decode([int(r["codepoint"], 16) for r in rows], fmt)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("args", FORMATS)
def test_gfloat_decode_and_boundaries(args):
    gf = pytest.importorskip("gfloat")
    from gfloat.formats import format_info_p3109
    fmt = P3109Format(*args)
    fi = format_info_p3109(fmt.k, fmt.precision,
                          gf.Signedness.Signed if fmt.signed else gf.Signedness.Unsigned,
                          gf.Domain.Extended if fmt.domain == "extended" else gf.Domain.Finite)
    codes = np.arange(1 << fmt.k, dtype=np.uint32)
    values = p3109_decode(codes, fmt)
    np.testing.assert_array_equal(values, gf.decode_ndarray(fi, codes))
    finite = np.sort(values[np.isfinite(values)])
    mid = finite[:-1] + (finite[1:] - finite[:-1]) / 2
    overflow_mid = fmt.max + np.ldexp(1.0, np.frexp(fmt.max)[1] - fmt.precision - 1)
    overflow_edges = np.array([np.nextafter(overflow_mid, -np.inf), overflow_mid,
                               np.nextafter(overflow_mid, np.inf)])
    if fmt.signed:
        overflow_edges = np.concatenate([overflow_edges, -overflow_edges])
    x = np.concatenate([overflow_edges, finite, mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf),
                        [fmt.max * 2, np.inf, np.nan, 0.0, -0.0]])
    reference_modes = [gf.RoundMode.TiesToEven, gf.RoundMode.TiesToAway, gf.RoundMode.TowardZero,
                       gf.RoundMode.TowardPositive, gf.RoundMode.TowardNegative]
    for mode, rnd in zip(MODES, reference_modes):
        for sat in [False, True]:
            actual = p3109_quantize(x, fmt, rounding=mode, saturate=sat)
            expected = gf.round_ndarray(fi, x, rnd, sat=sat)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{fmt.name}: {mode}, {sat}")
            assert not np.any(np.signbit(actual[actual == 0]))


@pytest.mark.parametrize("p", [1, 3, 4, 7])
@pytest.mark.parametrize("mode,reference", [("stochastic_a", "StochasticFastest"),
                                           ("stochastic_b", "StochasticFast"),
                                           ("stochastic_c", "Stochastic")])
def test_stochastic_reference(p, mode, reference):
    gf = pytest.importorskip("gfloat")
    from gfloat.formats import format_info_p3109
    rng = np.random.default_rng(82)
    x = rng.normal(size=2000) * 2
    bits = rng.integers(0, 256, size=x.shape, dtype=np.uint32)
    actual = p3109_quantize(x, P3109Format(8, p), rounding=mode, srbits=bits, srnumbits=8)
    expected = gf.round_ndarray(format_info_p3109(8, p), x, getattr(gf.RoundMode, reference),
                                srbits=bits, srnumbits=8)
    np.testing.assert_array_equal(actual, expected)


def test_specials_unsigned_and_saturation():
    f = P3109Format(signed=False)
    x = np.array([-np.inf, -1., -f.smallest_subnormal / 4, -0., 0., np.inf, np.nan])
    np.testing.assert_array_equal(p3109_quantize(x, f), [np.nan, np.nan, 0, 0, 0, np.inf, np.nan])
    for mode in ["toward_zero", "toward_positive"]:
        np.testing.assert_array_equal(p3109_quantize(x, f, rounding=mode),
                                      [np.nan, 0, 0, 0, 0, np.inf, np.nan])
    np.testing.assert_array_equal(p3109_quantize(x, f, saturate=True), [0, 0, 0, 0, 0, f.max, np.nan])
    np.testing.assert_array_equal(p3109_quantize(x, f, saturate="propagate"), [0, 0, 0, 0, 0, np.inf, np.nan])


def test_shapes_inputs_and_serialization():
    q = P3109(P3109Format(16, 11), saturate="propagate")
    restored = P3109.from_dict(json.loads(json.dumps(q.to_dict())))
    x = np.arange(30).reshape(5, 6)[:, ::2]
    x.setflags(write=False)
    np.testing.assert_array_equal(restored.decode(restored.encode(x)), q(x))
    assert q(1.1).shape == ()
    assert q([]).shape == (0,)
    assert q(np.empty((0, 3))).shape == (0, 3)
    assert not np.shares_memory(q(x), x)
    assert q(np.array([1], dtype=np.float32)).dtype == np.float64
    for value in [True, 1j, "1", np.array([object()])]:
        with pytest.raises(TypeError):
            q(value)
    for codes in [[-1], [65536], [1.2], [True]]:
        with pytest.raises((TypeError, ValueError)):
            q.decode(codes)


@pytest.mark.parametrize("kwargs", [{"k": 2}, {"k": True}, {"precision": 0}, {"precision": 8},
                                    {"k": 32, "precision": 1}, {"signed": 1}, {"domain": "bad"}])
def test_invalid_formats(kwargs):
    with pytest.raises((TypeError, ValueError)):
        P3109Format(**kwargs)


def test_to_odd_and_extreme_host_values():
    fmt = P3109Format()
    x = [1., 1.01, 1.125, 1.24, -1.01, 0., fmt.smallest_subnormal / 4]
    np.testing.assert_array_equal(p3109_quantize(x, fmt, rounding="to_odd"),
                                 [1., 1.125, 1.125, 1.125, -1.125, 0., fmt.smallest_subnormal])
    huge = np.array([np.finfo(float).max, -np.finfo(float).max])
    np.testing.assert_array_equal(p3109_quantize(huge, fmt, rounding="toward_zero"), [fmt.max, -fmt.max])
    for args in [(32, 23), (32, 32, False), (16, 7)]:
        f = P3109Format(*args)
        codes = np.random.default_rng(8).integers(0, 1 << f.k, 2000, dtype=np.uint64)
        np.testing.assert_array_equal(p3109_encode(p3109_decode(codes, f), f), codes)


def test_bad_policies_and_random_bits():
    for kwargs in [{"rounding": "bad"}, {"saturate": 1}, {"srnumbits": 3},
                   {"rounding": "stochastic_a"},
                   {"rounding": "stochastic_a", "srnumbits": 8, "srbits": [256]}]:
        with pytest.raises((TypeError, ValueError)):
            p3109_quantize([1.1], **kwargs)
