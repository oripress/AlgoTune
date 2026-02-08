import numpy as np
import scipy.fft
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """Compute the N-dimensional FFT using real FFT + reconstruction."""
        problem = np.asarray(problem, dtype=np.float64)
        
        if problem.ndim != 2:
            return scipy.fft.fftn(problem, workers=-1)
        
        n0, n1 = problem.shape
        
        # Use scipy's rfftn with parallel workers
        half = scipy.fft.rfftn(problem, workers=-1)
        nh = half.shape[1]  # n1 // 2 + 1
        
        if nh >= n1:
            return half
        
        # Number of columns to reconstruct
        nr = n1 - nh
        
        # Reconstruct full spectrum using Hermitian symmetry
        # result[k0, k1] = conj(half[(-k0) % n0, n1 - k1])
        result = np.empty((n0, n1), dtype=np.complex128)
        result[:, :nh] = half
        
        # Build the mirrored part efficiently
        # Row indices: [0, n0-1, n0-2, ..., 1] 
        # Col indices: [nr, nr-1, ..., 1] (which maps to half columns nr down to 1)
        
        # Using fancy indexing with pre-built arrays
        # half[row_mirror, col_mirror].conj()
        
        # Efficient reconstruction using slicing
        # First row is special (maps to row 0)
        # Remaining rows are reversed
        if n0 > 1:
            # half reversed rows [n0-1, n0-2, ..., 1, 0] then take cols [nr:0:-1]
            temp = np.conj(half[:, nr:0:-1])  # Reverse column order from nr to 1
            result[0, nh:] = temp[0]
            result[1:, nh:] = temp[-1:0:-1]  # Reverse rows 1 to n0-1
        else:
            result[0, nh:] = np.conj(half[0, nr:0:-1])
        
        return result