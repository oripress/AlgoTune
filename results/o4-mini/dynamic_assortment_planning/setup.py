from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("solver_hungarian.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)