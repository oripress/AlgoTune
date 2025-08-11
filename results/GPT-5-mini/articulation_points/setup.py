from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("art_cy", ["art_cy.pyx"]),
]

setup(
    name="art_cy",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)