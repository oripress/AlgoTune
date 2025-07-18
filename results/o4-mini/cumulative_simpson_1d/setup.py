from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("cum_simpson.pyx", compiler_directives={
        'boundscheck': False, 'wraparound': False, 'cdivision': True
    }),
    include_dirs=[np.get_include()],
)