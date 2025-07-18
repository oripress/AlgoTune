import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """Compute the 2D DCT Type I using scipy.fftpack."""
        import scipy.fftpack
        result = scipy.fftpack.dctn(problem, type=1)
        return result