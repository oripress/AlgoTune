# cython: language_level=3
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "solver_cy",
        ["solver_cy.pyx"],
        extra_compile_args=['-Ofast', '-march=native', '-funroll-loops', '-ffast-math'],
        extra_link_args=['-Ofast']
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True}),
)