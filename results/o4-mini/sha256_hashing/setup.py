from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "solver_cpp",
        sources=["solver_cpp.pyx"],
        libraries=["crypto"],  # link against OpenSSL libcrypto
        language="c",
    )
]

setup(
    name="solver_cpp",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)