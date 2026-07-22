import json
import os
import subprocess
import sys


def test_import_pychop_does_not_import_optional_backends():
    """Importing pychop should not eagerly import optional heavy frameworks."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()

    code = """
import json
import sys
import pychop

print(json.dumps({
    name: name in sys.modules
    for name in ["tensorflow", "torch", "jax", "jaxlib", "pandas"]
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    loaded = json.loads(proc.stdout)
    assert loaded == {
        "tensorflow": False,
        "torch": False,
        "jax": False,
        "jaxlib": False,
        "pandas": False,
    }
