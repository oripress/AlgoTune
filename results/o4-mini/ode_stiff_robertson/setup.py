from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="solver_c",
    ext_modules=cythonize("solver_c.pyx"),
    include_dirs=[np.get_include()],
)