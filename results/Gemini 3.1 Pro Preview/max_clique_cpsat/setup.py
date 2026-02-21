from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "fast_clique",
    sources=["fast_clique.pyx"],
    language="c++",
    extra_compile_args=["-O3", "-march=native", "-std=c++14"]
)

setup(
    ext_modules=cythonize([ext], compiler_directives={'language_level': "3"})
)