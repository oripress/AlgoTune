from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "_btsp_helpers",
        ["_btsp_helpers.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "_polymixed_helpers",
        ["_polymixed_helpers.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="algo_helpers",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)