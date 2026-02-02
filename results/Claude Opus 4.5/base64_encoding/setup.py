from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "base64_fast",
        ["base64_fast.pyx"],
        extra_compile_args=['-fopenmp', '-O3', '-march=native'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules, language_level=3),
    include_dirs=[np.get_include()]
)