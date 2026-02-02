from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "solver_cython",
        ["solver_cython.pyx"],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules, language_level=3),
    include_dirs=[numpy.get_include()]
)