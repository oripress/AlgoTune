from setuptools import Extension, setup

from Cython.Build import cythonize

extensions = [
    Extension(
        name="uf_cc",
        sources=["uf_cc.pyx"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="uf_cc",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
)