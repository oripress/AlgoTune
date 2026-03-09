from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        "fastproj",
        ["fastproj.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
    )
]

setup(
    name="fastproj",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
    ),
)