from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("hull", ["hull.pyx"], language="c++")
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[numpy.get_include()]
)