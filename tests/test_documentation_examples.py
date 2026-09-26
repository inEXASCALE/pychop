"""Execute core tutorial snippets independently so each example is self-contained."""
from pathlib import Path
import textwrap
import pytest

ROOT = Path(__file__).resolve().parents[1]


def snippets():
    for name in ("start", "p3109", "timeseries", "float_point"):
        lines = (ROOT / "docs" / "source" / (name + ".rst")).read_text().splitlines()
        for i, line in enumerate(lines):
            if line == ".. code-block:: python":
                block = []
                for candidate in lines[i + 1:]:
                    if candidate and not candidate.startswith("   "):
                        break
                    block.append(candidate)
                source = textwrap.dedent("\n".join(block)).strip()
                if any(import_line in source for import_line in
                       ["import torch", "import jax", "import tensorflow"]):
                    continue
                yield pytest.param(source, id=f"{name}-{i + 1}")


@pytest.mark.parametrize("source", list(snippets()))
def test_core_tutorial_snippet(source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(source, "<tutorial>", "exec"), {"__name__": "__tutorial__"})
