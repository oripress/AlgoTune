from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("sha256_fast.pyx", language_level=3)
)