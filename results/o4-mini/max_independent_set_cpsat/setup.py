from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("solver_c", ["solver_c.pyx"], extra_compile_args=["-O3"])
]

setup(
    name="solver_c",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)