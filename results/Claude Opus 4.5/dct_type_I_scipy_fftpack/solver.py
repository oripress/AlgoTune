import numpy as np
import scipy.fft

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the 2D DCT Type I using scipy.fft.
        """
        # Make a contiguous copy and allow overwrite with parallel workers
        x = np.array(problem, dtype=np.float64, order='C', copy=True)
        return scipy.fft.dctn(x, type=1, overwrite_x=True, workers=-1)