import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """Compute the 2D DST Type II."""
        import scipy.fftpack
        result = scipy.fftpack.dstn(problem, type=2)
        return result