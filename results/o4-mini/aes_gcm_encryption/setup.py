from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "aesgcm_c",
        sources=["aesgcm_c.pyx"],
        libraries=["crypto"],
    ),
]

setup(
    name="aesgcm_c",
    ext_modules=cythonize(extensions, language_level="3"),
)