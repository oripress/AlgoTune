from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        Extension(
            "solver_cython",
            ["solver_cython.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-march=native"]
        ),
        language_level=3
    ),
)
# Trigger recompilation