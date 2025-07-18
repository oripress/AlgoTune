from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "domset",
        ["domset.pyx"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="domset",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)