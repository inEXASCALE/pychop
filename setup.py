import setuptools
import platform
import logging


def get_version(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ = '"):
                return line.split("'")[1]
    raise RuntimeError('Error in parsing version string.')
    
PRJECT_NAME = "pychop"
PACKAGE_NAME = "pychop"
VERSION = get_version('pychop/__init__.py')
SETREQUIRES=["numpy"]
MAINTAINER="Erin Carson, Xinye Chen"
EMAIL="xinyechenai@gmail.com"
INREUIRES=["numpy>=1.17.2", "pandas", "dask[array]", "torch>=1.12", "jax>=0.4.8", "jaxlib>=0.4.7"]

AUTHORS="Erin Carson, Xinye Chen"

with open("README.md", 'r') as f:
    long_description = f.read()

ext_errors = (ModuleNotFoundError, IOError, SystemExit)
logging.basicConfig()
log = logging.getLogger(__file__)

if platform.python_implementation() == "PyPy":
    NUMPY_MIN_VERSION = "1.19.2"
else:
    NUMPY_MIN_VERSION = "1.17.2"
   

from setuptools.command.build_ext import build_ext
    
class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)

metadata = {"name":PRJECT_NAME,
            'packages':["pychop", "pychop.np", "pychop.tch", "pychop.jx"],
            "version":VERSION,
            "setup_requires":SETREQUIRES,
            "install_requires":INREUIRES,
            'cmdclass': {'build_ext': CustomBuildExtCommand},
            "long_description":long_description,
            "author":AUTHORS,
            "maintainer":MAINTAINER,
            "author_email":EMAIL,
            "classifiers":[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3"
            ],
            "maintainer_email":EMAIL,
            "description":"Python code for simulating low precision floating-point arithmetic",
            "long_description_content_type":'text/markdown',
            "url":"https://github.com/nla-group/pychop.git",
            "license":'MIT License'
}
            

class InvalidVersion(ValueError):
    """raise invalid version error"""

    
if __name__ == "__main__":
    try:
        setuptools.setup(
            **metadata
        )
    except ext_errors as ext:
        log.warning(ext)
        log.warning("failure Installation.")
