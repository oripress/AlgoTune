from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "hodgkin_huxley_cython",
        ["hodgkin_huxley_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3']
    )
]

setup(
    name="hodgkin_huxley_cython",
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)