import numpy as np
from scipy.fftpack import dct
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute DCT Type I with axis-wise dct and overwrite optimization.
        """
        result = np.ascontiguousarray(problem, dtype=np.float64)
        result = dct(result, type=1, axis=0, overwrite_x=True)
        result = dct(result, type=1, axis=1, overwrite_x=True)
        return result