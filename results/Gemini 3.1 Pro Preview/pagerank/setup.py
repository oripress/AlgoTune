from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("pagerank.pyx", compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False, 'cdivision': True}),
    include_dirs=[np.get_include()]
)