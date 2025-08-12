from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        Extension(
            "l0prune",
            ["l0prune.pyx"],
            include_dirs=[np.get_include()],
        )
    )
)