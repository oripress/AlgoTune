from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="cy_wasserstein",
    ext_modules=cythonize("cy_wasserstein.pyx", annotate=False),
    include_dirs=[np.get_include()],
)