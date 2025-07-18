from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="l1proj",
    ext_modules=cythonize(
        "l1proj.pyx",
        compiler_directives={'boundscheck': False, 'cdivision': True, 'wraparound': False}
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)