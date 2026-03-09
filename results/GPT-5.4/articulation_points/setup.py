from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        name="articulation_cy",
        sources=["articulation_cy.pyx"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    )
)