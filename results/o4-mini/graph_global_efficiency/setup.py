from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("solver_cy.pyx", annotate=False),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("solver_cy.pyx"),
    zip_safe=False,
)