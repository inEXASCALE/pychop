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

# Core dependencies - lightweight and stable
install_requires = [
    "numpy>=1.17.3",
    "pandas",
    "scipy>=1.0",
    "scikit-learn>=0.20",
    "dask[array]",
]

# Optional heavy backends as extras to prevent installation conflicts
# (especially protobuf/tensorflow issues on macOS/Anaconda)
extras_require = {
    "torch": ["torch>=1.12"],
    "jax": ["jax>=0.4.8", "jaxlib>=0.4.7"],
    "tensorflow": ["tensorflow>=2.16"],
    "all": [
        "torch>=1.12",
        "jax>=0.4.8",
        "jaxlib>=0.4.7",
        "tensorflow>=2.16",
    ],
}


class CustomBuildExtCommand(build_ext):
    """Custom build_ext command to add numpy headers when needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super().run()


if __name__ == "__main__":
    setuptools.setup(
        version=VERSION,
        install_requires=install_requires,
        extras_require=extras_require,
        packages=setuptools.find_packages(),
        cmdclass={"build_ext": CustomBuildExtCommand},
    )