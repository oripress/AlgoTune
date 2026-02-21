import numpy as np
import scipy.fft
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        return scipy.fft.dstn(problem.astype(np.float32, copy=False), type=2, overwrite_x=True, workers=64)