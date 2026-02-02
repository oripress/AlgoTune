from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

# Check for OpenMP support (simplified check)
# Usually -fopenmp for GCC/Clang
extra_compile_args = ['-fopenmp', '-O3']
extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        "fast_solver",
        ["fast_solver.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(extensions),
)