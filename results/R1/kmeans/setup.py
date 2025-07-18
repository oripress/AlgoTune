from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fast_kmeans.pyx"),
    include_dirs=[np.get_include()]
)