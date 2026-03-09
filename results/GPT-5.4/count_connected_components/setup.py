from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "ccsolver",
        ["ccsolver.pyx"],
    )
]

setup(
    name="ccsolver",
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