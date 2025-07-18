import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """Optimized DCT Type I with parallel processing"""
        return dctn(problem, type=1, workers=-1, overwrite_x=True)