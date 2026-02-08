from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "sha256_fast",
    sources=["sha256_fast.pyx"],
    libraries=["crypto"],
)

setup(
    ext_modules=cythonize([ext], language_level=3),
)