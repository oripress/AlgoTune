from setuptools import setup
from Cython.Build import cythonize

setup(
    name="fastchacha",
    ext_modules=cythonize("fastchacha.pyx", compiler_directives={'language_level': "3"}),
    zip_safe=False,
)