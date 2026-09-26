import json

import numpy as np
import pytest

from pychop import SymbolicTimeSeries


def test_known_paa_and_quantiles():
    model = SymbolicTimeSeries(segment_size=2, alphabet_size=2, normalize=False)
    x = np.array([0., 2., 4., 6.])
    np.testing.assert_array_equal(model.fit_transform(x), [0, 1])
    assert model.to_dict()["breakpoints"] == [3.0]
    np.testing.assert_array_equal(model.inverse_transform([0, 1]), [1, 1, 5, 5])
    assert model.transform([3, 3])[0] == 1  # boundary goes to upper bin


def test_export_roundtrip_and_no_refit():
    rng = np.random.default_rng(9)
    train, test = rng.normal(size=(10, 32)), rng.normal(size=(3, 32)) + 2
    model = SymbolicTimeSeries(4, 8).fit(train)
    params = model.to_dict()
    restored = SymbolicTimeSeries.from_dict(json.loads(json.dumps(params)))
    np.testing.assert_array_equal(model.transform(test), restored.transform(test))
    assert model.to_dict() == params
    assert restored.inverse_transform(restored.transform(test)).shape == test.shape
    params["centers"][0] = -10000
    assert model.to_dict() != params


def test_constant_and_noncontiguous():
    model = SymbolicTimeSeries(2, 4)
    x = np.ones((2, 16))[:, ::2]
    x.setflags(write=False)
    symbols = model.fit_transform(x)
    np.testing.assert_array_equal(model.inverse_transform(symbols), x)
    assert model.to_dict()["scale"] == 1
    assert symbols.dtype == np.uint8


def test_invalid_input_and_atomic_fit():
    model = SymbolicTimeSeries(2, 4)
    with pytest.raises(RuntimeError):
        model.transform([1, 2])
    model.fit([1, 2, 3, 4])
    before = model.to_dict()
    for x in [[], [1, 2, 3], [np.nan, 0], [np.inf, 0], 1, [1e308, -1e308]]:
        with pytest.raises(ValueError):
            model.fit(x)
    assert model.to_dict() == before
    for symbols in [[-1], [4], 1, [1.2]]:
        with pytest.raises((TypeError, ValueError)):
            model.inverse_transform(symbols)
    for key, value in [("scale", 0), ("centers", [1]), ("schema_version", 2),
                       ("breakpoints", [2, 1, 0]), ("mean", float("nan"))]:
        bad = dict(before, **{key: value})
        with pytest.raises(ValueError):
            SymbolicTimeSeries.from_dict(bad)


@pytest.mark.parametrize("kwargs", [{"segment_size": 0}, {"segment_size": True},
                                    {"alphabet_size": 257}, {"normalize": 1}])
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SymbolicTimeSeries(**kwargs)
