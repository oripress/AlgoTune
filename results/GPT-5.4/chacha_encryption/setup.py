from setuptools import Extension, setup

from Cython.Build import cythonize

extensions = [
    Extension(
        "fastchacha",
        ["fastchacha.pyx"],
        libraries=["crypto"],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)