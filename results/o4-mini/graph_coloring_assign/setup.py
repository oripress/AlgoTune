from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("dsatur", ["dsatur.pyx"])
]

setup(
    name="dsatur",
    ext_modules=cythonize(extensions, language_level="3"),
)