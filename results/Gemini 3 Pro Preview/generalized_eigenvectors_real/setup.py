from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["fast_solver_float.pyx"]),
    include_dirs=[np.get_include()]
)