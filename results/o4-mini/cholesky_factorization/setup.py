from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("solver_chol", ["solver_chol.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name="solver_chol",
    ext_modules=cythonize(extensions, language_level=3),
)