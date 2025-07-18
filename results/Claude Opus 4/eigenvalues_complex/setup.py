from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "eigenvalue_solver",
        ["eigenvalue_solver.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)