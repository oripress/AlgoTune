from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "fast_solver",
        ["fast_solver.pyx"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"]
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)