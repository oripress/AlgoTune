from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "sha256_fast",
        ["sha256_fast.pyx"],
        libraries=["crypto"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    )
)