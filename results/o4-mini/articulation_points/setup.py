from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "solver_cy",
    ["solver_cy.pyx"],
    extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops", "-flto"],
    extra_link_args=["-flto"]
)

setup(
    ext_modules=cythonize([ext], compiler_directives={'boundscheck': False, 'wraparound': False})
)