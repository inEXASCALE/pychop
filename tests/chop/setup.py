from setuptools import setup, Extension
import pybind11
import os
import sys

# Ensure CUDA is available
CUDA_PATH = os.environ.get('CUDA_PATH', '/usr/local/cuda')  # Adjust this if CUDA is elsewhere
if not os.path.exists(CUDA_PATH):
    raise RuntimeError("CUDA not found. Please set CUDA_PATH environment variable or adjust the path.")

# Define the extension module
binary_float_module = Extension(
    'binary_float_cuda',  # Name of the resulting Python module
    sources=['binary_float_binding.cpp', 'binary_float.cu'],
    include_dirs=[
        pybind11.get_include(),  # Pybind11 headers
        os.path.join(CUDA_PATH, 'include'),  # CUDA headers
    ],
    library_dirs=[os.path.join(CUDA_PATH, 'lib64')],  # CUDA libraries
    libraries=['cudart', 'curand'],  # Link against CUDA runtime and cuRAND
    extra_compile_args={
        'cxx': ['-O3', '-fPIC'],  # C++ flags
        'nvcc': ['-O3', '--use_fast_math', '-Xcompiler', '-fPIC'],  # CUDA flags
    },
    language='c++',
)

# Custom build extension to handle CUDA compilation
from setuptools.command.build_ext import build_ext

class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Override compiler for .cu files
        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or {}
            for source in ext.sources:
                if source.endswith('.cu'):
                    # Use nvcc for CUDA files
                    self.compiler.set_executable('compiler_so', 'nvcc')
                    self.compiler.set_executable('linker_so', 'nvcc -shared')
                else:
                    # Use default C++ compiler for .cpp files
                    self.compiler.set_executable('compiler_so', self.compiler.compiler_so[0])
                    self.compiler.set_executable('linker_so', self.compiler.linker_so[0])
        build_ext.build_extensions(self)

# Setup configuration
setup(
    name='binary_float_cuda',
    version='0.1',
    description='CUDA-accelerated binary floating-point simulator',
    ext_modules=[binary_float_module],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=['pybind11>=2.6', 'numpy'],
    zip_safe=False,
)