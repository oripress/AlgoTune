from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='linear_solver',
    ext_modules=cythonize("solver.pyx"),
    include_dirs=[np.get_include()],
)