from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "hh_solver.pyx",
        compiler_directives={'boundscheck': False, 'wraparound': False, 'cdivision': True},
    )
)