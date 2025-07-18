from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# Define the Cython extension
ext_modules = [
    Extension(
        "solver_cy",
        ["solver_cy.pyx"],
        include_dirs=[numpy.get_include()], # Add numpy headers
        extra_compile_args=["-O3"], # Enable optimizations
        extra_link_args=["-O3"],
    )
]

# Setup configuration
setup(
    name="solver_cy",
    ext_modules=cythonize(ext_modules, language_level="3", annotate=False),
)