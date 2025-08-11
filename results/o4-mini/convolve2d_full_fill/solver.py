import os
import psutil

# Determine number of physical cores for threading
_workers = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
# Set environment variables to control threading in FFT and BLAS libraries
os.environ["OMP_NUM_THREADS"] = str(_workers)
os.environ["OPENBLAS_NUM_THREADS"] = str(_workers)
os.environ["MKL_NUM_THREADS"] = str(_workers)
os.environ["NUMEXPR_NUM_THREADS"] = str(_workers)

import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the 2D convolution using FFT in single precision for speed,
        then convert result back to float64 for compatibility.
        """
        a, b = problem
        # Convert to float32 for faster FFT operations
        a32 = np.asarray(a, dtype=np.float32)
        b32 = np.asarray(b, dtype=np.float32)
        # Perform convolution in float32
        c32 = fftconvolve(a32, b32, mode='full')
        # Cast result back to float64
        return c32.astype(np.float64)