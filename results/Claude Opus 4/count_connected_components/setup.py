from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "uf_cython",
        ["uf_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'initializedcheck': False,
    })
)