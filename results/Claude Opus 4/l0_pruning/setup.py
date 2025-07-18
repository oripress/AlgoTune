from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("l0_solver.pyx"),
    include_dirs=[numpy.get_include()]
)