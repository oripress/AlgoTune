from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            Extension(
                "simplex_cy",
                ["simplex_cy.pyx"],
                include_dirs=[np.get_include()],
            )
        ],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)