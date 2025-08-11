from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="articulation",
    ext_modules=cythonize(
        "articulation.pyx",
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        },
        annotate=False,
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)