from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="seirs_solver",
        sources=["seirs_solver.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="seirs_solver",
    ext_modules=cythonize(extensions, language_level="3"),
)