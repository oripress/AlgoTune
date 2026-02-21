from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("kruskal_cython.pyx", compiler_directives={'language_level': "3"}),
)