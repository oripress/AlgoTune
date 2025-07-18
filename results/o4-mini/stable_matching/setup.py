from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="solver_ext",
        sources=["solver_ext.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="solver_ext",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)