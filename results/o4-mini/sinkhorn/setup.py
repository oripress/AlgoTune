from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "sinkhorn_cython.pyx",
        compiler_directives={'boundscheck': False, 'wraparound': False, 'nonecheck': False, 'cdivision': True}
    ),
    include_dirs=[numpy.get_include()],
)