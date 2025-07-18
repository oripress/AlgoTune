from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="dijkstra_cy",
        sources=["dijkstra_cy.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    )
]

setup(
    name="dijkstra_cy",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)