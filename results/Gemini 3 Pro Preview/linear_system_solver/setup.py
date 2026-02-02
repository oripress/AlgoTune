from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("solver_impl.pyx"),
    include_dirs=[numpy.get_include()]
)