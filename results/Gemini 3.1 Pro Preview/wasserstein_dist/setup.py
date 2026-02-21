from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "solver_cython",
        ["solver_cython.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
)