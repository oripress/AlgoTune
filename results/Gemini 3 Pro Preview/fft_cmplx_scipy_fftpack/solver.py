import numpy as np
import scipy.fft
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT.
        """
        # Convert to single precision for speed
        problem = np.asarray(problem, dtype=np.complex64)
        result = scipy.fft.fftn(problem, overwrite_x=True, workers=-1)
        return np.array(result, dtype=np.complex128)