from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("b64encoder", ["b64encoder.pyx"])
]

setup(
    name="b64encoder",
    ext_modules=cythonize(extensions, language_level="3"),
)
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("b64encoder.pyx", language_level="3"),
    zip_safe=False,
)