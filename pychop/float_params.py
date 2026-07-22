"""Format parameter tables for pychop floating-point-like formats.

This module keeps the historical ``float_params`` API for scalar binary
floating-point formats and extends the summary table to pychop's predefined
block formats: BFP, Flexpoint, and OCP MX.
"""

# This API follows https://github.com/higham/chop/blob/master/float_params.m

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

pd = None


def _import_pandas():
    """Import pandas only when a DataFrame result is requested."""
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd
    return pd


@dataclass(frozen=True)
class _BinaryFormat:
    canonical: str
    aliases: Tuple[str, ...]
    p: int
    exp_bits: int
    emax: int
    description: str


_BINARY_FORMATS: Tuple[_BinaryFormat, ...] = (
    _BinaryFormat(
        canonical="q43",
        aliases=("q43", "fp8-e4m3"),
        p=4,
        exp_bits=4,
        emax=7,
        description="FP8 E4M3 quarter precision",
    ),
    _BinaryFormat(
        canonical="q52",
        aliases=("q52", "fp8-e5m2"),
        p=3,
        exp_bits=5,
        emax=15,
        description="FP8 E5M2 quarter precision",
    ),
    _BinaryFormat(
        canonical="bfloat16",
        aliases=("b", "bfloat16", "bf16"),
        p=8,
        exp_bits=8,
        emax=127,
        description="bfloat16",
    ),
    _BinaryFormat(
        canonical="fp16",
        aliases=("h", "half", "fp16"),
        p=11,
        exp_bits=5,
        emax=15,
        description="IEEE half precision",
    ),
    _BinaryFormat(
        canonical="tf32",
        aliases=("t", "tf32"),
        p=11,
        exp_bits=8,
        emax=127,
        description="NVIDIA TensorFloat-32",
    ),
    _BinaryFormat(
        canonical="fp32",
        aliases=("s", "single", "fp32"),
        p=24,
        exp_bits=8,
        emax=127,
        description="IEEE single precision",
    ),
    _BinaryFormat(
        canonical="fp64",
        aliases=("d", "double", "fp64"),
        p=53,
        exp_bits=11,
        emax=1023,
        description="IEEE double precision",
    ),
    _BinaryFormat(
        canonical="fp128",
        aliases=("q", "quadruple", "fp128"),
        p=113,
        exp_bits=15,
        emax=16383,
        description="IEEE quadruple precision",
    ),
)

_BINARY_FORMAT_BY_ALIAS: Mapping[str, _BinaryFormat] = {
    alias: fmt for fmt in _BINARY_FORMATS for alias in fmt.aliases
}

_TABLE_COLUMNS = [
    "",
    "family",
    "u",
    "xmins",
    "xmin",
    "xmax",
    "p",
    "emins",
    "emin",
    "emax",
    "element_bits",
    "block_size",
    "scale_bits",
    "aliases",
    "description",
]


def binary_mark(value: Any) -> str:
    """Represent a value as an approximate power of two.

    Parameters
    ----------
    value : scalar
        Value to format. Finite positive powers of two are represented as
        ``"2^k"``. Values that cannot be converted to a finite base-two
        exponent are returned with ``str(value)``.

    Returns
    -------
    mark : str
        Human-readable base-two representation.

    Examples
    --------
    >>> binary_mark(0.5)
    '2^-1'
    >>> binary_mark(float("inf"))
    'inf'
    """
    try:
        value_float = float(value)
        if not np.isfinite(value_float) or value_float <= 0:
            return str(value)
        exp = int(np.round(np.log2(value_float)))
        return "2^" + str(exp)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _binary_tuple(fmt: _BinaryFormat) -> Tuple[float, float, float, float, int, int, int, int]:
    """Compute scalar binary floating-point parameters.

    Parameters
    ----------
    fmt : _BinaryFormat
        Binary format registry entry.

    Returns
    -------
    params : tuple of float and int
        Tuple ``(u, xmins, xmin, xmax, p, emins, emin, emax)`` used by the
        historical ``float_params`` API.
    """
    emin = 1 - fmt.emax
    emins = emin + 1 - fmt.p
    xmins = 2.0 ** emins
    xmin = 2.0 ** emin

    try:
        xmax = 2.0 ** fmt.emax * (2.0 - 2.0 ** (1 - fmt.p))
    except OverflowError:
        xmax = float("inf")

    u = 2.0 ** (-fmt.p)
    return u, xmins, xmin, xmax, fmt.p, emins, emin, fmt.emax


def _format_numeric(value: Optional[float], binary: bool) -> str:
    """Format a numeric table cell.

    Parameters
    ----------
    value : float or None
        Value to render. ``None`` is rendered as ``"N/A"``.
    binary : bool
        Whether finite values should be rendered as approximate powers of two.

    Returns
    -------
    text : str
        Formatted table cell.
    """
    if value is None:
        return "N/A"
    if binary:
        return binary_mark(value)
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:9.2e}"


def _format_int(value: Optional[int]) -> str:
    """Format an integer table cell.

    Parameters
    ----------
    value : int or None
        Integer value to render. ``None`` is rendered as ``"N/A"``.

    Returns
    -------
    text : str
        Formatted table cell.
    """
    return "N/A" if value is None else f"{value:d}"


def _row(
    name: str,
    family: str,
    params: Mapping[str, Any],
    binary: bool,
    aliases: Iterable[str] = (),
    description: str = "",
) -> Dict[str, str]:
    """Build one normalized parameter table row.

    Parameters
    ----------
    name : str
        Canonical format name shown in the first table column.
    family : str
        Format family, such as ``"binary"``, ``"bfp"``, or ``"mx"``.
    params : mapping
        Named format parameters.
    binary : bool
        Whether numeric fields should use base-two notation.
    aliases : iterable of str, default=()
        Supported aliases for the format.
    description : str, default=""
        Human-readable format description.

    Returns
    -------
    row : dict of str
        Row compatible with the ``float_params(None)`` DataFrame schema.
    """
    return {
        "": name,
        "family": family,
        "u": _format_numeric(params.get("u"), binary),
        "xmins": _format_numeric(params.get("xmins"), binary),
        "xmin": _format_numeric(params.get("xmin"), binary),
        "xmax": _format_numeric(params.get("xmax"), binary),
        "p": _format_int(params.get("p")),
        "emins": _format_int(params.get("emins")),
        "emin": _format_int(params.get("emin")),
        "emax": _format_int(params.get("emax")),
        "element_bits": _format_int(params.get("element_bits")),
        "block_size": _format_int(params.get("block_size")),
        "scale_bits": _format_int(params.get("scale_bits")),
        "aliases": ", ".join(aliases),
        "description": description,
    }


def _binary_params_dict(fmt: _BinaryFormat) -> Dict[str, Any]:
    """Return scalar binary parameters as a named dictionary.

    Parameters
    ----------
    fmt : _BinaryFormat
        Binary format registry entry.

    Returns
    -------
    params : dict
        Named parameter dictionary used to build the summary table.
    """
    u, xmins, xmin, xmax, p, emins, emin, emax = _binary_tuple(fmt)
    return {
        "family": "binary",
        "canonical": fmt.canonical,
        "aliases": fmt.aliases,
        "u": u,
        "xmins": xmins,
        "xmin": xmin,
        "xmax": xmax,
        "p": p,
        "emins": emins,
        "emin": emin,
        "emax": emax,
        "element_bits": 1 + fmt.exp_bits + (p - 1),
        "description": fmt.description,
    }


def _mx_params_dict(spec: Any) -> Dict[str, Any]:
    """Return representative global range parameters for an OCP MX format.

    Parameters
    ----------
    spec : MXSpec
        OCP MX format specification.

    Returns
    -------
    params : dict
        Named parameter dictionary containing representative scalar ranges and
        MX block metadata.
    """
    try:
        from .np.mx_formats import _mx_float_params, _scale_exponent_bounds
    except ImportError:
        from pychop.np.mx_formats import _mx_float_params, _scale_exponent_bounds

    element_params = _mx_float_params(spec)
    min_scale_exp, max_scale_exp = _scale_exponent_bounds(spec.scale_exp_bits)
    max_value = element_params["max"] * (2.0 ** max_scale_exp)
    min_positive = element_params.get("smallest_subnormal", 2.0 ** (-spec.sig_bits))
    normal_positive = element_params.get("smallest_normal", min_positive)

    if spec.exp_bits == 0:
        p = spec.element_bits - 1
    else:
        p = spec.sig_bits + 1

    return {
        "family": "mx",
        "canonical": spec.name.lower(),
        "u": 2.0 ** (-p),
        "xmins": min_positive * (2.0 ** min_scale_exp),
        "xmin": normal_positive * (2.0 ** min_scale_exp),
        "xmax": max_value,
        "p": p,
        "emins": int(math.floor(math.log2(min_positive)) + min_scale_exp),
        "emin": int(math.floor(math.log2(normal_positive)) + min_scale_exp),
        "emax": int(math.floor(math.log2(max_value))),
        "element_bits": spec.element_bits,
        "block_size": spec.block_size,
        "scale_bits": spec.scale_exp_bits + spec.scale_sig_bits,
        "description": f"OCP MX {spec.name} with E{spec.scale_exp_bits}M{spec.scale_sig_bits} scale",
    }


def _bfp_params_dict(spec: Any) -> Dict[str, Any]:
    """Return representative global range parameters for a BFP format.

    Parameters
    ----------
    spec : BFPSpec
        Block floating point format specification.

    Returns
    -------
    params : dict
        Named parameter dictionary containing representative scalar ranges and
        BFP block metadata.
    """
    min_exp, max_exp = _shared_exponent_bounds(spec.exponent_bits)
    mantissa_levels = 2 ** (spec.mantissa_bits - 1)
    max_int = mantissa_levels - 1
    max_normalized = max_int / mantissa_levels
    quantum_exp = min_exp - (spec.mantissa_bits - 1)

    return {
        "family": "bfp",
        "canonical": spec.name.lower(),
        "u": 2.0 ** (-spec.mantissa_bits),
        "xmins": 2.0 ** quantum_exp,
        "xmin": 2.0 ** quantum_exp,
        "xmax": max_normalized * (2.0 ** max_exp),
        "p": spec.mantissa_bits,
        "emins": quantum_exp,
        "emin": quantum_exp,
        "emax": int(math.floor(math.log2(max_normalized * (2.0 ** max_exp)))),
        "element_bits": spec.mantissa_bits,
        "block_size": spec.block_size,
        "scale_bits": spec.exponent_bits,
        "description": f"BFP with {spec.mantissa_bits}-bit signed mantissas",
    }


def _shared_exponent_bounds(exponent_bits: int) -> Tuple[int, int]:
    """Return exponent limits for a biased unsigned exponent field.

    Parameters
    ----------
    exponent_bits : int
        Number of bits in the shared exponent field.

    Returns
    -------
    min_exponent : int
        Smallest representable unbiased exponent.
    max_exponent : int
        Largest finite representable unbiased exponent.
    """
    bias = 2 ** (exponent_bits - 1) - 1
    return -bias, (2 ** exponent_bits - 2) - bias


def _all_format_rows(binary: bool) -> pd.DataFrame:
    """Build the full pychop-supported format table.

    Parameters
    ----------
    binary : bool
        Whether numeric fields should use base-two notation.

    Returns
    -------
    table : pandas.DataFrame
        Table containing scalar binary formats and registered BFP/MX formats.
    """
    rows = []

    for fmt in _BINARY_FORMATS:
        rows.append(
            _row(
                fmt.canonical,
                "binary",
                _binary_params_dict(fmt),
                binary=binary,
                aliases=fmt.aliases,
                description=fmt.description,
            )
        )

    BFP_FORMATS, MX_FORMATS = _format_registries()

    for name, spec in BFP_FORMATS.items():
        rows.append(
            _row(
                name,
                "bfp",
                _bfp_params_dict(spec),
                binary=binary,
                aliases=(name,),
                description=f"Block floating point ({spec.name})",
            )
        )

    for name, spec in MX_FORMATS.items():
        rows.append(
            _row(
                name,
                "mx",
                _mx_params_dict(spec),
                binary=binary,
                aliases=(name,),
                description=f"OCP microscaling ({spec.name})",
            )
        )

    pandas = _import_pandas()
    return pandas.DataFrame(rows, columns=_TABLE_COLUMNS)


def float_params(prec: Optional[str] = None, binary: bool = False, *argv: Any):
    """Return parameter information for pychop-supported formats.

    Parameters
    ----------
    prec : str, default=None
        Format name or alias. If ``None``, return a table containing all
        predefined scalar and block formats known to pychop.

        Scalar binary aliases include ``"q43"``, ``"fp8-e4m3"``, ``"q52"``,
        ``"fp8-e5m2"``, ``"b"``, ``"bf16"``, ``"bfloat16"``, ``"h"``,
        ``"half"``, ``"fp16"``, ``"t"``, ``"tf32"``, ``"s"``, ``"single"``,
        ``"fp32"``, ``"d"``, ``"double"``, ``"fp64"``, ``"q"``,
        ``"quadruple"``, and ``"fp128"``.

        Block format names include entries from :data:`pychop.BFP_FORMATS`
        and :data:`pychop.MX_FORMATS`, such as ``"bfp8"``,
        ``"flexpoint16"``, ``"mxfp8_e4m3"``, ``"mxfp4_e2m1"``, and
        ``"mxint8"``.
    binary : bool, default=False
        If ``True`` and ``prec`` is ``None``, display numeric table entries as
        powers of two where possible. This option does not change individual
        query return values.
    *argv : tuple
        Ignored positional arguments kept for backward compatibility with the
        original MATLAB-inspired API.

    Returns
    -------
    params : pandas.DataFrame or tuple or dict
        If ``prec`` is ``None``, a :class:`pandas.DataFrame` with one row per
        supported format is returned.

        If ``prec`` names a scalar binary floating-point format, the historical
        tuple ``(u, xmins, xmin, xmax, p, emins, emin, emax)`` is returned.

        If ``prec`` names a predefined BFP, Flexpoint, or OCP MX block format,
        a dictionary with named fields is returned. Block formats use shared
        exponent or shared scale metadata, so their parameters are not exactly
        the same object as the scalar binary tuple.

    See Also
    --------
    pychop.BFP_FORMATS : Predefined block floating point formats.
    pychop.MX_FORMATS : Predefined OCP microscaling formats.

    Notes
    -----
    For BFP and MX formats, ``xmins``, ``xmin``, and ``xmax`` are
    representative global ranges induced by the predefined block metadata.
    Actual quantization still operates block by block.

    Examples
    --------
    >>> u, xmins, xmin, xmax, p, emins, emin, emax = float_params("fp16")
    >>> p
    11
    >>> float_params("mxfp8_e4m3")["family"]
    'mx'
    >>> table = float_params()
    >>> "mxint8" in set(table[""])
    True
    """
    if prec is None:
        return _all_format_rows(binary=binary)

    key = str(prec).lower()
    if key in _BINARY_FORMAT_BY_ALIAS:
        return _binary_tuple(_BINARY_FORMAT_BY_ALIAS[key])

    BFP_FORMATS, MX_FORMATS = _format_registries()

    if key in BFP_FORMATS:
        return _bfp_params_dict(BFP_FORMATS[key])

    if key in MX_FORMATS:
        return _mx_params_dict(MX_FORMATS[key])

    supported = sorted(
        set(_BINARY_FORMAT_BY_ALIAS)
        | set(BFP_FORMATS)
        | set(MX_FORMATS)
    )
    raise ValueError(
        "Unsupported precision format. Supported values include: "
        + ", ".join(supported)
    )


def _format_registries() -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return registered BFP and MX format dictionaries.

    Returns
    -------
    bfp_formats : mapping
        Registry of predefined BFP and Flexpoint formats.
    mx_formats : mapping
        Registry of predefined OCP MX formats.
    """
    try:
        from .bfp_formats import BFP_FORMATS
        from .mx_formats import MX_FORMATS
    except ImportError:
        from pychop.bfp_formats import BFP_FORMATS
        from pychop.mx_formats import MX_FORMATS

    return BFP_FORMATS, MX_FORMATS
