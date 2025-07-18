from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="solver_cy",
    ext_modules=cythonize(
        "solver_cy.pyx",
        annotate=False,
        language_level="3"
    ),
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)