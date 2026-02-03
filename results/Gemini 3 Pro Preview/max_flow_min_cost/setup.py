from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "solver_cython",
        ["solver_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        extra_link_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(extensions),
)