from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("base64_cython.pyx"),
)
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("base64_cython.pyx"),
)
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "base64_cython.pyx",
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'nonecheck': False,
        },
        annotate=True,
    ),
)
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("base64_cython.pyx")
)