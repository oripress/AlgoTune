import numpy as np
from scipy.fft import fftn

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the N-dimensional FFT of a real-valued input array
        using SciPy's optimized pocketfft implementation.
        """
        # Ensure a contiguous float64 array
        arr = np.asarray(problem, dtype=np.float64, order='C')
        # Directly invoke pocketfft FFTN
        return fftn(arr)