# Contributing

Contributions to code, tests, examples and documentation are welcome. Open an
issue to discuss a proposed change, or submit a pull request with a description
of its behavior and how it was verified.

## Numerical changes

Preserve input ownership, validate format boundaries and cover numerical edge
cases with focused tests. Optional framework imports must remain lazy so a core
installation works without Torch, TensorFlow or JAX. Tests for optional backends
should skip cleanly when the framework is unavailable.

## Checks

Run the tests relevant to your change from the repository root. For a full run:

```bash
python -m pip install -e . pytest
python -m pytest tests -q
```

`tests/test_import_laziness.py` checks that importing `pychop` in a fresh Python
process does not import TensorFlow, Torch, JAX, jaxlib or pandas.

To check the documentation, install `docs/requirements.txt` and run:

```bash
python -m sphinx -b html -W --keep-going docs/source /tmp/pychop-docs
```
