from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("uf_cython.pyx", compiler_directives={'boundscheck': False, 'wraparound': False, 'cdivision': True}),
    include_dirs=[np.get_include()],
)