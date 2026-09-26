"""Calibrated piecewise-aggregate time-series symbolization.

Fit normalization and empirical quantile bins on training series, then reuse
those parameters unchanged for validation, test, and deployment data. This is
an empirical SAX-style representation, not symbolic algebra or a predictor.
"""
import numpy as np

from ._validation import bounded_integer as _integer

__all__ = ["SymbolicTimeSeries"]


class SymbolicTimeSeries:
    """Convert (..., time) arrays to integer symbols using non-overlapping PAA.

    ``segment_size`` must divide the time dimension exactly. ``fit`` learns a
    global training mean/scale and quantile breakpoints from segment means.
    Constant training data use scale 1. Missing/infinite data are rejected.
    ``inverse_transform`` repeats learned bin centers; reconstruction is lossy.
    """

    def __init__(self, segment_size=8, alphabet_size=8, normalize=True):
        self._segment_size = _integer(segment_size, "segment_size", 1, 2**31 - 1)
        self._alphabet_size = _integer(alphabet_size, "alphabet_size", 2, 256)
        if type(normalize) is not bool:
            raise TypeError("normalize must be a bool")
        self._normalize = normalize
        self._state = None

    @property
    def segment_size(self):
        return self._segment_size

    @property
    def alphabet_size(self):
        return self._alphabet_size

    @property
    def normalize(self):
        return self._normalize

    def _series(self, values):
        raw = np.asarray(values)
        if raw.dtype.kind not in "fiu":
            raise TypeError("series must be real numeric array-like data")
        x = raw.astype(np.float64, copy=False)
        if x.ndim == 0 or x.size == 0:
            raise ValueError("series must have a nonempty time axis")
        if x.shape[-1] % self.segment_size:
            raise ValueError("time dimension must be divisible by segment_size")
        if not np.all(np.isfinite(x)):
            raise ValueError("series must contain only finite values")
        return x

    def _paa(self, x, mean, scale):
        with np.errstate(over="ignore", invalid="ignore"):
            z = (x - mean) / scale
            result = z.reshape(*x.shape[:-1], -1, self.segment_size).mean(axis=-1)
        if not np.all(np.isfinite(result)):
            raise ValueError("series magnitude exceeds stable float64 normalization")
        return result

    def fit(self, values):
        """Learn calibration from training series and return self.

        Failed calibration leaves any previously fitted state intact.
        """
        x = self._series(values)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = float(x.mean()) if self.normalize else 0.0
            scale = float(x.std()) if self.normalize else 1.0
        if not np.isfinite(mean) or not np.isfinite(scale):
            raise ValueError("training magnitude exceeds stable float64 calibration")
        scale = scale if scale > 0 else 1.0
        paa = self._paa(x, mean, scale).ravel()
        cuts = np.quantile(paa, np.arange(1, self.alphabet_size) / self.alphabet_size)
        labels = np.searchsorted(cuts, paa, side="right")
        # Quantile midpoints provide deterministic representatives for empty bins.
        centers = np.quantile(paa, (np.arange(self.alphabet_size) + .5) / self.alphabet_size)
        for i in range(self.alphabet_size):
            members = paa[labels == i]
            if members.size:
                centers[i] = members.mean()
        if not np.all(np.isfinite(cuts)) or not np.all(np.isfinite(centers)):
            raise ValueError("training magnitude exceeds stable float64 bin calibration")
        self._state = (mean, scale, cuts, centers)
        return self

    def _fitted(self):
        if self._state is None:
            raise RuntimeError("fit the symbolizer or load parameters before transforming")
        return self._state

    def transform(self, values):
        """Return uint8 symbols in [0, alphabet_size), preserving leading axes."""
        mean, scale, cuts, _ = self._fitted()
        paa = self._paa(self._series(values), mean, scale)
        return np.searchsorted(cuts, paa, side="right").astype(np.uint8)

    def fit_transform(self, values):
        """Fit on training data and return its symbols."""
        return self.fit(values).transform(values)

    def inverse_transform(self, symbols):
        """Reconstruct an approximation in the original training units."""
        mean, scale, _, centers = self._fitted()
        codes = np.asarray(symbols)
        if codes.dtype.kind not in "iu":
            raise TypeError("symbols must contain integers")
        if codes.ndim == 0 or np.any(codes < 0) or np.any(codes >= self.alphabet_size):
            raise ValueError("symbols need a time axis and must lie in the fitted alphabet")
        return np.repeat(centers[codes] * scale + mean, self.segment_size, axis=-1)

    def to_dict(self):
        """Export the complete fitted state as a JSON-compatible versioned mapping."""
        mean, scale, cuts, centers = self._fitted()
        return {"schema_version": 1, "segment_size": self.segment_size,
                "alphabet_size": self.alphabet_size, "normalize": self.normalize,
                "mean": mean, "scale": scale, "breakpoints": cuts.tolist(),
                "centers": centers.tolist()}

    @classmethod
    def from_dict(cls, parameters):
        """Restore and validate exported state without fitting on new data."""
        expected = {"schema_version", "segment_size", "alphabet_size", "normalize",
                    "mean", "scale", "breakpoints", "centers"}
        if set(parameters) != expected:
            raise ValueError("invalid symbolizer parameter fields")
        if type(parameters["schema_version"]) is not int or parameters["schema_version"] != 1:
            raise ValueError("unsupported symbolizer schema_version")
        obj = cls(parameters["segment_size"], parameters["alphabet_size"], parameters["normalize"])
        mean, scale = float(parameters["mean"]), float(parameters["scale"])
        cuts = np.array(parameters["breakpoints"], dtype=np.float64, copy=True)
        centers = np.array(parameters["centers"], dtype=np.float64, copy=True)
        if (not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0
                or cuts.shape != (obj.alphabet_size - 1,) or centers.shape != (obj.alphabet_size,)
                or not np.all(np.isfinite(cuts)) or not np.all(np.isfinite(centers))
                or np.any(np.diff(cuts) < 0) or np.any(np.diff(centers) < 0)):
            raise ValueError("invalid symbolizer calibration")
        if not obj.normalize and (mean != 0 or scale != 1):
            raise ValueError("unnormalized symbolizer requires mean=0 and scale=1")
        obj._state = (mean, scale, cuts, centers)
        return obj
