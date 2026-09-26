"""Keep process-global backend configuration from leaking between tests."""
import pytest


@pytest.fixture(autouse=True)
def isolated_backend(monkeypatch):
    monkeypatch.setenv("chop_backend", "auto")
