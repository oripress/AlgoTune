from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "fast_base64",
        ["fast_base64.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="fast_base64",
    ext_modules=cythonize(extensions),
)