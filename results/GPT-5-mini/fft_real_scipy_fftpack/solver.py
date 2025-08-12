import numpy as np
import scipy.fftpack as fftpack
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        a = np.asarray(problem)
        return fftpack.fftn(a)