import numpy as np
import scipy.fft
from numpy.typing import NDArray
from typing import Any
import numba
from numba import jit

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """
        Compute the N-dimensional DCT Type I using optimized approach.
        """
        # For small arrays, use direct scipy approach
        if problem.size < 1000:
            return scipy.fft.dctn(problem, type=1, overwrite_x=True)
        else:
            # For larger arrays, use optimized approach
            x = np.asarray(problem, dtype=np.float64)
            return scipy.fft.dctn(x, type=1, overwrite_x=True)