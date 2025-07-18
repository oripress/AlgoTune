from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("solver_cy.pyx", compiler_directives={'boundscheck': False, 'wraparound': False}),
    include_dirs=[np.get_include()],
)