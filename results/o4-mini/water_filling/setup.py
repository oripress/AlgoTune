from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="waterfill_ext",
        sources=["waterfill_ext.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    )
]

setup(
    name="waterfill_ext",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)