import numpy as np
from pychop import Chop


def test_auto_backend_reuses_rng_and_matches_explicit(monkeypatch):
    x = np.full(2000, 1.1)
    monkeypatch.setenv("chop_backend", "auto")
    auto = Chop(5, 2, rmode=5, random_state=71)
    a1 = auto(x)
    impl = auto._impl
    a2 = auto(x)
    assert auto._impl is impl
    assert np.any(a1 != a2)
    monkeypatch.setenv("chop_backend", "numpy")
    explicit = Chop(5, 2, rmode=5, random_state=71)
    np.testing.assert_array_equal(a1, explicit(x))
    np.testing.assert_array_equal(a2, explicit(x))


def test_auto_scalar_and_list_fallback(monkeypatch):
    monkeypatch.setenv("chop_backend", "auto")
    q = Chop(5, 10)
    for x in [1.1, np.float64(1.1), [1.1, 2.2], (1.1, 2.2)]:
        np.testing.assert_array_equal(q(x), q(np.asarray(x)))
