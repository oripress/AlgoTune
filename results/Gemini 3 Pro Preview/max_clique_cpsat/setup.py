from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        Extension(
            "clique_solver",
            sources=["clique_solver.pyx"],
            language="c++",
            extra_compile_args=["-std=c++11", "-O3"]
        ),
        language_level=3
    )
)