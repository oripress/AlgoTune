from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="gs",
        sources=["gs.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        extra_link_args=[],
        language="c",
    )
]

setup(
    name="gs",
    ext_modules=cythonize(extensions, language_level=3, annotate=False),
    zip_safe=False,
)