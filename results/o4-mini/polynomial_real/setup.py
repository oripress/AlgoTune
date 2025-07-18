from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="solver_cy",
    ext_modules=cythonize("solver_cy.pyx", annotate=False),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)