import numpy as np
from numpy.typing import NDArray
import scipy.fft

class Solver:
    def __init__(self):
        # Pre-configure FFT workers for better performance
        scipy.fft.set_workers(1)  # Single-threaded for small arrays
    
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute 2D DCT Type I using scipy.fft with optimizations.
        """
        # Ensure contiguous memory layout for better cache performance
        if not problem.flags['C_CONTIGUOUS']:
            problem = np.ascontiguousarray(problem)
        
        # Use scipy.fft.dctn which uses pocketfft backend
        return scipy.fft.dctn(problem, type=1, norm=None, axes=None, overwrite_x=False)