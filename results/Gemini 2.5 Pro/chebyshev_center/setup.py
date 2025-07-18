from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module with specific compiler and linker flags
# for maximum performance.
extensions = [
    Extension(
        "helper",  # name of the resulting module
        ["helper.pyx"],  # list of source files
        include_dirs=[numpy.get_include()],
        # -O3: Aggressive optimization
        # -ffast-math: Allow optimizations that may violate strict IEEE standards
        # -march=native: Optimize for the specific CPU architecture of the build machine (enables AVX, etc.)
        # -fopenmp: Enable OpenMP for parallelization (for prange)
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

# The 'cythonize' function handles the conversion from .pyx to .c,
# and setuptools handles the compilation of the .c file into a shared library.
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        # Enable annotation to generate a .html file for performance analysis.
        annotate=True
    )
)