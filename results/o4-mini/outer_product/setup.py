from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "solver_ext.pyx",
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'language_level': 3
        }
    ),
    include_dirs=[numpy.get_include()],
)