from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("graph_mis", ["graph_mis.pyx"], extra_compile_args=["-O3"])
]

setup(
    name="graph_mis",
    ext_modules=cythonize(extensions, compiler_directives={'language_level':"3"})
)