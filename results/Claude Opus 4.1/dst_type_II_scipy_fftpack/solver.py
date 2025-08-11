import numpy as np
from scipy import fft
from numpy.typing import NDArray
from typing import Any

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """Compute 2D DST Type II using scipy.fft which is often faster."""
        # Use scipy.fft.dstn which can be faster than scipy.fftpack.dstn
        # Ensure contiguous array for better cache performance
        if not problem.flags['C_CONTIGUOUS']:
            problem = np.ascontiguousarray(problem)
        
        # Use scipy.fft.dstn with type 2
        result = fft.dstn(problem, type=2, workers=-1)  # workers=-1 uses all available CPUs
        return result