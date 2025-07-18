from setuptools import setup
from Cython.Build import cythonize

setup(
    name='solver_c',
    ext_modules=cythonize("solver_c.pyx", annotate=False, language_level=3),
    zip_safe=False,
)