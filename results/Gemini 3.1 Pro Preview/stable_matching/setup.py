from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "fast_solver.pyx",
        compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False}
    )
)