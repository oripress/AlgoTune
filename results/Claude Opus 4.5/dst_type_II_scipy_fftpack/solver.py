import numpy as np
import scipy.fft
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """
        Compute the 2D DST Type II by applying 1D DST along each axis.
        """
        # Apply DST along axis 1 first, then axis 0 with overwrite
        temp = scipy.fft.dst(problem, type=2, axis=1, workers=-1)
        result = scipy.fft.dst(temp, type=2, axis=0, workers=-1, overwrite_x=True)
        return result