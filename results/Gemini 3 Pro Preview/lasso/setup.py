from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("lasso_cython.pyx"),
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops"]
)