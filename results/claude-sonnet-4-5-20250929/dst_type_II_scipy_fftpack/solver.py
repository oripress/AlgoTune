import numpy as np
from numpy.typing import NDArray
from scipy.fft import dstn

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute N-dimensional DST Type II using scipy.fft.dstn (newer interface).
        """
        result = dstn(problem, type=2, norm='backward')
        return result