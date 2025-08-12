from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("solver_cy", ["solver_cy.pyx"], include_dirs=[np.get_include()]),
]

setup(
    name="solver_cy",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)