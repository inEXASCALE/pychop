# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Type, Callable, Iterator, Tuple

import array_api_compat
import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest

from gfloat import RoundMode, decode_float, decode_ndarray, round_float, round_ndarray
from gfloat.formats import *

from gfloat.round_ndarray import _rnito, _rnite


@pytest.mark.parametrize("int_type", [np.int64, np.int16])
def test_rnito_rnite(int_type: npt.DTypeLike) -> None:

    xp = array_api_compat.array_namespace(np.array(0.0))

    np.testing.assert_equal(_rnite(xp.array(3.5), int_type), 4.0)
    np.testing.assert_equal(_rnite(xp.array(2.5), int_type), 2.0)
    np.testing.assert_equal(_rnite(xp.array(0.5), int_type), 0.0)
    np.testing.assert_equal(_rnite(xp.array(-0.5), int_type), 0.0)
    np.testing.assert_equal(_rnite(xp.array(-2.5), int_type), -2.0)
    np.testing.assert_equal(_rnite(xp.array(-3.5), int_type), -4.0)

    np.testing.assert_equal(_rnito(xp.array(3.5), int_type), 3.0)
    np.testing.assert_equal(_rnito(xp.array(2.5), int_type), 3.0)
    np.testing.assert_equal(_rnito(xp.array(0.5), int_type), 1.0)
    np.testing.assert_equal(_rnito(xp.array(-0.5), int_type), -1.0)
    np.testing.assert_equal(_rnito(xp.array(-3.5), int_type), -3.0)
    np.testing.assert_equal(_rnito(xp.array(-2.5), int_type), -3.0)


def rnd_scalar(
    fi: FormatInfo, v: float, mode: RoundMode = RoundMode.TiesToEven, sat: bool = False
) -> float:
    return round_float(fi, v, mode, sat)


def rnd_array(
    fi: FormatInfo, v: float, mode: RoundMode = RoundMode.TiesToEven, sat: bool = False
) -> float:
    a = round_ndarray(fi, np.asarray([v]), mode, sat)
    return float(a[0])


@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_p3109(round_float: Callable) -> None:
    fi = format_info_p3109(8, 4)
    assert round_float(fi, 0.0068359375) == 0.0068359375
    assert round_float(fi, 0.0029296875) == 0.0029296875
    assert round_float(fi, 0.0078125) == 0.0078125
    assert round_float(fi, 0.017578125) == 0.017578125
    assert round_float(fi, 224.0) == 224.0
    assert round_float(fi, 240.0) == np.inf

    assert round_float(fi, 224.1, RoundMode.TowardPositive) == np.inf

    assert round_float(fi, 232.0) == 224.0
    assert round_float(fi, 232.0, RoundMode.TiesToAway) == np.inf
    assert round_float(fi, 232.0, RoundMode.TowardZero) == 224.0
    assert round_float(fi, 232.0, RoundMode.TowardNegative) == 224.0
    assert round_float(fi, 232.0, RoundMode.TowardPositive) == np.inf

    assert round_float(fi, -232.0) == -224.0
    assert round_float(fi, -232.0, RoundMode.TiesToAway) == -np.inf
    assert round_float(fi, -232.0, RoundMode.TowardZero) == -224.0
    assert round_float(fi, -232.0, RoundMode.TowardNegative) == -np.inf
    assert round_float(fi, -232.0, RoundMode.TowardPositive) == -224.0

    assert round_float(fi, 232.1) == np.inf


p4min = 2**-10  # smallest subnormal in 8p4


@pytest.mark.parametrize(
    "mode, vals",
    (
        (
            RoundMode.TowardZero,
            (
                (p4min, p4min),
                (p4min / 4, 0.0),
                (p4min / 2, 0.0),
                (-p4min, -p4min),
                (-p4min / 4, 0.0),
                (-p4min / 2, 0.0),
                (64.0, 64.0),
                (63.0, 60.0),
                (62.0, 60.0),
                (-64.0, -64.0),
                (-63.0, -60.0),
                (-62.0, -60.0),
            ),
        ),
        (
            RoundMode.TowardPositive,
            (
                (p4min, p4min),
                (p4min / 4, p4min),
                (p4min / 2, p4min),
                (-p4min, -p4min),
                (-p4min / 4, 0.0),
                (-p4min / 2, 0.0),
                (64.0, 64.0),
                (63.0, 64.0),
                (62.0, 64.0),
                (-64.0, -64.0),
                (-63.0, -60.0),
                (-62.0, -60.0),
            ),
        ),
        (
            RoundMode.TowardNegative,
            (
                (p4min, p4min),
                (p4min / 4, 0.0),
                (p4min / 2, 0.0),
                (-p4min, -p4min),
                (-p4min / 4, -p4min),
                (-p4min / 2, -p4min),
                (64.0, 64.0),
                (63.0, 60.0),
                (62.0, 60.0),
                (-64.0, -64.0),
                (-63.0, -64.0),
                (-62.0, -64.0),
            ),
        ),
        (
            RoundMode.TiesToEven,
            (
                (p4min, p4min),
                (p4min / 4, 0.0),
                (p4min / 2, 0.0),
                (-p4min, -p4min),
                (-p4min / 4, 0.0),
                (-p4min / 2, 0.0),
                (64.0, 64.0),
                (63.0, 64.0),
                (62.0, 64.0),
                (61.0, 60.0),
                (-64.0, -64.0),
                (-63.0, -64.0),
                (-62.0, -64.0),
                (-61.0, -60.0),
                (-58.0, -56.0),
            ),
        ),
        (
            RoundMode.TiesToAway,
            (
                (p4min, p4min),
                (p4min / 4, 0.0),
                (p4min / 2, p4min),
                (-p4min, -p4min),
                (-p4min / 4, 0.0),
                (-p4min / 2, -p4min),
                (64.0, 64.0),
                (63.0, 64.0),
                (62.0, 64.0),
                (61.0, 60.0),
                (-64.0, -64.0),
                (-63.0, -64.0),
                (-62.0, -64.0),
                (-61.0, -60.0),
                (-58.0, -60.0),
            ),
        ),
    ),
)
@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_p3109b(round_float: Callable, mode: RoundMode, vals: list) -> None:
    fi = format_info_p3109(8, 4)

    for sat in (True, False):
        for val, expected in vals:
            assert round_float(fi, val, mode, sat) == expected


p4max = 224.0
p4maxup = 240.0
p4maxhalfup = (p4max + p4maxup) / 2


@pytest.mark.parametrize(
    "modesat, vals",
    (
        (
            (RoundMode.TowardZero, True),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, p4max),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -p4max),
            ),
        ),
        (
            (RoundMode.TowardZero, False),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, np.inf),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -np.inf),
            ),
        ),
        (
            (RoundMode.TowardPositive, True),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, p4max),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -p4max),
            ),
        ),
        (
            (RoundMode.TowardPositive, False),
            (
                (p4max, p4max),
                (p4maxhalfup, np.inf),
                (p4maxup, np.inf),
                (np.inf, np.inf),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -np.inf),
            ),
        ),
        (
            (RoundMode.TowardNegative, True),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, p4max),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -p4max),
            ),
        ),
        (
            (RoundMode.TowardNegative, False),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, np.inf),
                (-p4max, -p4max),
                (-p4maxhalfup, -np.inf),
                (-p4maxup, -np.inf),
                (-np.inf, -np.inf),
            ),
        ),
        (
            (RoundMode.TiesToEven, True),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, p4max),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -p4max),
            ),
        ),
        (
            (RoundMode.TiesToEven, False),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, np.inf),
                (np.inf, np.inf),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -np.inf),
                (-np.inf, -np.inf),
            ),
        ),
        (
            (RoundMode.TiesToAway, True),
            (
                (p4max, p4max),
                (p4maxhalfup, p4max),
                (p4maxup, p4max),
                (np.inf, p4max),
                (-p4max, -p4max),
                (-p4maxhalfup, -p4max),
                (-p4maxup, -p4max),
                (-np.inf, -p4max),
            ),
        ),
        (
            (RoundMode.TiesToAway, False),
            (
                (p4max, p4max),
                (p4maxhalfup, np.inf),
                (p4maxup, np.inf),
                (np.inf, np.inf),
                (-p4max, -p4max),
                (-p4maxhalfup, -np.inf),
                (-p4maxup, -np.inf),
                (-np.inf, -np.inf),
            ),
        ),
    ),
    ids=lambda x: f"{str(x[0])}-{'Sat' if x[1] else 'Inf'}" if len(x) == 2 else None,
)
@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_p3109_sat(
    round_float: Callable, modesat: tuple[RoundMode, bool], vals: list
) -> None:
    fi = format_info_p3109(8, 4)

    for val, expected in vals:
        assert round_float(fi, val, *modesat) == expected


@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_e5m2(round_float: Callable) -> None:
    fi = format_info_ocp_e5m2

    assert fi.max == 57344

    assert round_float(fi, 1.5258789e-05) == 2**-16

    # Default NONSAT rounding
    assert round_float(fi, 57344.0) == 57344
    assert round_float(fi, 57344.1) == 57344
    assert round_float(fi, 61439.9) == 57344
    assert round_float(fi, 61440.0) == np.inf
    assert round_float(fi, np.inf, sat=False) == np.inf
    assert round_float(fi, -np.inf, sat=False) == -np.inf
    assert np.isnan(round_float(fi, np.nan, sat=False))

    # SAT rounding
    assert round_float(fi, 57344.0, sat=True) == 57344
    assert round_float(fi, 57344.1, sat=True) == 57344
    assert round_float(fi, 61439.9, sat=True) == 57344
    assert round_float(fi, 61440.0, sat=True) == 57344
    assert round_float(fi, np.inf, sat=True) == 57344
    assert round_float(fi, -np.inf, sat=True) == -57344
    assert np.isnan(round_float(fi, np.nan, sat=True))


@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_e4m3(round_float: Callable) -> None:
    fi = format_info_ocp_e4m3

    assert fi.max == 448

    # Default NONSAT rounding
    assert round_float(fi, 448.0) == 448
    assert round_float(fi, 448.1) == 448
    assert round_float(fi, 464.0) == 448
    assert np.isnan(round_float(fi, 464.01))
    assert np.isnan(round_float(fi, np.inf, sat=False))
    assert np.isnan(round_float(fi, -np.inf, sat=False))
    assert np.isnan(round_float(fi, np.nan, sat=False))

    # SAT rounding
    assert round_float(fi, 448.0, sat=True) == 448
    assert round_float(fi, 448.1, sat=True) == 448
    assert round_float(fi, 464.0, sat=True) == 448
    assert round_float(fi, 464.01, sat=True) == 448
    assert round_float(fi, np.inf, sat=True) == 448
    assert round_float(fi, -np.inf, sat=True) == -448
    assert np.isnan(round_float(fi, np.nan, sat=True))


some_positive_codepoints = (
    0x00,
    0x01,
    0x02,
    0x03,
    0x07,
    0x0F,
    0x17,
    0x21,
    0x33,
    0x40,
    0x53,
    0x65,
    0x70,
)


@pytest.mark.parametrize(
    "fi",
    [
        format_info_ocp_e5m2,
        format_info_ocp_e4m3,
        *p3109_binary8_formats,
    ],
)
def test_round(fi: FormatInfo) -> None:
    """
    Test rounding from values between exact binary8 values
    For integer code point i, let
      v0 = the float value at i
      v1 = the float value at i+1, i.e. nextUp(v0)
      dv = v1 - v0
    Then check that:
        round(v0) == v0
        round(v0 + 0.3*dv) == v0
        round(v0 + 0.6*dv) == v1
    """

    def get_vals() -> Iterator[Tuple[float, float]]:
        for i in some_positive_codepoints:
            v0 = decode_float(fi, i + 0).fval
            v1 = decode_float(fi, i + 1).fval
            if np.isfinite([v0, v1]).all():
                dv = v1 - v0
                nearest_even = v0 if (i & 1 == 0) else v1
                yield v0, v0
                yield v0 + 0.3 * dv, v0
                yield v0 + 0.49 * dv, v0
                yield v0 + 0.51 * dv, v1
                yield v0 + 0.99 * dv, v1
                yield v0 + 0.50 * dv, nearest_even

    for v, expected in get_vals():
        assert round_float(fi, v) == expected

    vs = np.array([v for v, _ in get_vals()])
    expecteds = np.array([expected for _, expected in get_vals()])

    got = round_ndarray(fi, vs)
    np.testing.assert_equal(got, expecteds)


test_formats = [
    (format_info_ocp_e5m2, ml_dtypes.float8_e5m2),
    (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn),
]


def _linterp(a, b, t):  # type: ignore[no-untyped-def]
    return a * (1 - t) + b * t


def _mlround(v: float, dty: Type) -> float:
    """
    Round `v` using ml_dtypes library
    """
    return np.array([v]).astype(dty).astype(float).item()


@pytest.mark.parametrize("fi,mldtype", test_formats)
@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_ml_dtype_compatible(
    round_float: Callable, fi: FormatInfo, mldtype: Type
) -> None:
    """
    Test that rounding is compatible with ml_dtypes
    """
    for i in range(255):
        # For each float v, check values at various interpolations
        # between v and nextUp(v)
        v0 = decode_float(fi, i + 0).fval
        v1 = decode_float(fi, i + 1).fval

        for alpha in (0, 0.3, 0.5, 0.6, 0.9, 1.25):
            v = _linterp(v0, v1, alpha)
            if np.isfinite(v):
                val = round_float(fi, v, RoundMode.TiesToEven)

                mlval = _mlround(v, mldtype)
                np.testing.assert_equal(val, mlval)


@pytest.mark.parametrize("fi,mldtype", test_formats)
@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_ints(round_float: Callable, fi: FormatInfo, mldtype: Type) -> None:
    for v in np.arange(289).astype(float):
        val = round_float(fi, v)

        mlval = _mlround(v, mldtype)
        np.testing.assert_equal(val, mlval)


@pytest.mark.parametrize("fi", sample_formats)
@pytest.mark.parametrize("round_float", (rnd_scalar, rnd_array))
def test_round_roundtrip(round_float: Callable, fi: FormatInfo) -> None:
    if fi.bits <= 8:
        step = 1
    elif fi.bits <= 16:
        step = 13
    elif fi.bits <= 32:
        step = 73013
    elif fi.bits <= 64:
        step = (73013 << 32) + 39

    for i in range(0, 2**fi.bits, step):
        fv = decode_float(fi, i)
        fval2 = round_float(fi, fv.fval)
        np.testing.assert_equal(fval2, fv.fval)


@pytest.mark.parametrize(
    "v, srnumbits, expected_up",
    (
        (259, 3, 0.0 / 8),
        (259, 5, 2.0 / 32),
        (277, 3, 3.0 / 8),
        (288, 3, 0.5),
        (311, 3, 7.0 / 8),
    ),
)
@pytest.mark.parametrize("impl", ("scalar", "array"))
def test_stochastic_rounding(
    impl: bool, v: float, srnumbits: int, expected_up: float
) -> None:
    fi = format_info_ocp_e5m2

    v0 = round_float(fi, v, RoundMode.TowardNegative)
    v1 = round_float(fi, v, RoundMode.TowardPositive)

    n = 10_000
    expected_up_count = expected_up * n

    srbits = np.random.randint(0, 2**srnumbits, size=(n,))
    if impl == "scalar":
        count_v1 = 0
        for k in range(n):
            r = round_float(
                fi,
                v,
                RoundMode.Stochastic,
                sat=False,
                srbits=srbits[k],
                srnumbits=srnumbits,
            )
            if r == v1:
                count_v1 += 1
            else:
                assert r == v0
    else:
        vs = np.full(n, v)
        rs = round_ndarray(fi, vs, RoundMode.Stochastic, False, srbits, srnumbits)
        assert np.all((rs == v0) | (rs == v1))
        count_v1 = np.sum(rs == v1)

    print(f"SRBits={srnumbits}, observed = {count_v1}, expected = {expected_up_count} ")
    # e.g. if expected is 1250/10000, want to be within 0.75,1.25
    # this is loose, but should still catch logic errors
    atol = n * 2.0 ** (-1 - srnumbits)
    np.testing.assert_allclose(count_v1, expected_up_count, atol=atol / 2)


@pytest.mark.parametrize(
    "rnd",
    (
        RoundMode.Stochastic,
        RoundMode.StochasticOdd,
        RoundMode.StochasticFast,
        RoundMode.StochasticFastest,
    ),
)
@pytest.mark.parametrize("srnumbits", [3, 8, 9, 16, 32])
@pytest.mark.parametrize("sat", (True, False))
def test_stochastic_rounding_scalar_eq_array(
    rnd: RoundMode, srnumbits: int, sat: bool
) -> None:
    fi = format_info_ocp_e5m2

    v0 = decode_ndarray(fi, np.arange(255))
    v1 = decode_ndarray(fi, np.arange(255) + 1)
    ok = np.isfinite(v0) & np.isfinite(v1)
    v0 = v0[ok]
    v1 = v1[ok]

    for alpha in (0, 0.3, 0.5, 0.6, 0.7, 0.9, 1.25):
        v = _linterp(v0, v1, alpha)
        assert np.isfinite(v).all()
        srbits = np.random.randint(0, 2**srnumbits, v.shape)

        val_array = round_ndarray(
            fi,
            v,
            rnd,
            sat=sat,
            srbits=srbits,
            srnumbits=srnumbits,
        )

        val_scalar = [
            round_float(
                fi,
                vi,
                rnd,
                sat=sat,
                srbits=srbitsi,
                srnumbits=srnumbits,
            )
            for vi, srbitsi in zip(v, srbits)
        ]

        np.testing.assert_equal(val_array, val_scalar)

        # Ensure faithful rounding
        if alpha < 1.0:
            assert ((val_array == v0) | (val_array == v1)).all()


def test_large_bfloat() -> None:
    # from https://github.com/graphcore-research/gfloat/pull/49

    a = 6.6461399789245764e35
    b = 6.620178494631905e35

    assert b < a
    rounded_a = round_float(format_info_bfloat16, a, RoundMode.TowardZero)
    rounded_b = round_float(format_info_bfloat16, b, RoundMode.TowardZero)

    assert rounded_b <= rounded_a

    rounded_a_array = round_ndarray(
        format_info_bfloat16, np.array([a]), RoundMode.TowardZero
    )
    rounded_b_array = round_ndarray(
        format_info_bfloat16, np.array([b]), RoundMode.TowardZero
    )

    assert all(rounded_b_array <= rounded_a_array)
