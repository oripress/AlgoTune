from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "_power_control_cython",
        sources=["_power_control_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=['-O3']
    )
]

setup(
    name="power_control_optimizer",
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)