import platform
import setuptools
from setuptools.command.build_ext import build_ext


def get_version(fname: str) -> str:
    """Read the __version__ string from __init__.py."""
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ ="):
                return line.split("'")[1].strip()
    raise RuntimeError("Unable to find __version__ string.")


VERSION = get_version("pychop/__init__.py")

# Dynamically set numpy version requirement based on interpreter (CPython vs PyPy)
if platform.python_implementation() == "PyPy":
    numpy_req = "numpy>=1.19.2"
else:
    numpy_req = "numpy>=1.17.2"

install_requires = [
    numpy_req,
    "pandas",
    "dask[array]",
    "torch>=1.12",
    "jax>=0.4.8",
    "jaxlib>=0.4.7",
]

class CustomBuildExtCommand(build_ext):
    """Custom build_ext command to add numpy headers when needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super().run()


if __name__ == "__main__":
    setuptools.setup(
        version=VERSION,
        install_requires=install_requires,  # Corresponds to dynamic dependencies in pyproject.toml
        packages=setuptools.find_packages(),  # Automatically discover all packages and subpackages
        cmdclass={"build_ext": CustomBuildExtCommand},  
    )