import numpy as np
from numpy.typing import NDArray

class Solver:
    def __init__(self):
        """Initialize the solver with precomputed FFT plans."""
        import scipy.fftpack
        self.scipy_fftpack = scipy.fftpack
        
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional FFT using scipy.fftpack with optimized data types.
        
        Args:
            problem: Input real-valued n×n matrix
            **kwargs: Additional keyword arguments
            
        Returns:
            Complex n×n matrix representing the FFT result
        """
        # Convert to numpy array with float32 for better performance
        problem_array = np.asarray(problem, dtype=np.float32)
        
        # Use scipy.fftpack with optimized data type
        return self.scipy_fftpack.fftn(problem_array)