# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import pytest

import numpy.typing as npt
import torch

from gfloat import FormatInfo, RoundMode, round_ndarray
from gfloat.formats import format_info_ocp_e5m2, format_info_p3109


def test_torch() -> None:
    """
    Test that Torch tensors agree with e5m2
    """
    a = torch.randn(1024)

    a8 = a.to(dtype=torch.float8_e5m2).to(dtype=torch.float32)

    fi = format_info_ocp_e5m2
    t8 = round_ndarray(fi, a)  # type: ignore [arg-type]

    torch.testing.assert_close(a8, t8, atol=0.0, rtol=0.0)

    # Check torch.compile
    tc = torch.compile(lambda x: round_ndarray(fi, x))
    t8i = tc(a)

    torch.testing.assert_close(a8, t8i, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "rnd",
    (
        RoundMode.TowardZero,
        RoundMode.TowardNegative,
        RoundMode.TowardPositive,
        RoundMode.TiesToEven,
        RoundMode.TiesToAway,
    ),
)
@pytest.mark.parametrize("fi", (format_info_ocp_e5m2, format_info_p3109(8, 3)))
@pytest.mark.parametrize("sat", (True, False))
def test_torch_compile_agrees(fi: FormatInfo, rnd: RoundMode, sat: bool) -> None:
    """
    Test that Torch compile output agrees with eager
    """
    a = torch.randn(1024)
    a[18] = torch.inf
    a[19] = -torch.inf

    t8 = round_ndarray(fi, a, rnd, sat)  # type: ignore [arg-type]

    # Check torch.compile
    tc = torch.compile(lambda x: round_ndarray(fi, x, rnd, sat))
    t8i = tc(a)

    torch.testing.assert_close(t8, t8i, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "rnd",
    (
        RoundMode.Stochastic,
        RoundMode.StochasticOdd,
        RoundMode.StochasticFast,
        RoundMode.StochasticFastest,
    ),
)
@pytest.mark.parametrize("fi", (format_info_ocp_e5m2, format_info_p3109(8, 3)))
def test_torch_compile_agrees_sr(fi: FormatInfo, rnd: RoundMode) -> None:
    """
    Test that Torch tensors don't crash
    """
    a = torch.randn(1024)
    a[18] = torch.inf
    a[19] = -torch.inf

    srnumbits = 5
    srbits = torch.randint(0, 2**srnumbits, a.shape)

    t8 = round_ndarray(fi, a, rnd, srbits=srbits, srnumbits=srnumbits)  # type: ignore [arg-type]

    # Check torch.compile
    @torch.compile
    def tc(x: npt.NDArray) -> npt.NDArray:
        return round_ndarray(fi, x, rnd, srbits=srbits, srnumbits=srnumbits)  # type: ignore [arg-type]

    t8_tc = tc(a)  # type: ignore [arg-type]

    torch.testing.assert_close(t8, t8_tc, atol=0.0, rtol=0.0)

    # Check torch.compile dynamic
    @torch.compile(dynamic=True)
    def tc2(x: npt.NDArray) -> npt.NDArray:
        return round_ndarray(fi, x, rnd, srbits=srbits, srnumbits=srnumbits)  # type: ignore [arg-type]

    t8_tc2 = tc2(a)  # type: ignore [arg-type]

    torch.testing.assert_close(t8, t8_tc2, atol=0.0, rtol=0.0)
