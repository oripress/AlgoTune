from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("solver_core.pyx", language_level=3),
)