import scipy.fft
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using scipy.fft with parallel workers.
        """
        return scipy.fft.fftn(problem, workers=-1, norm="backward")