import numpy as np
import scipy.fft
import scipy.fftpack
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using scipy.fft.
        """
        if problem.size < 10000:
             return scipy.fftpack.dctn(problem, type=1, overwrite_x=True)
        
        return scipy.fft.dctn(problem, type=1, workers=-1, overwrite_x=True)