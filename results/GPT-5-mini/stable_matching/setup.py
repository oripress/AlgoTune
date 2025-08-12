from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("cy_gs", ["cy_gs.pyx"]),
]

setup(
    name="cy_gs",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)