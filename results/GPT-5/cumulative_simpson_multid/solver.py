from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson as _cumulative_simpson

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array
        using SciPy's cumulative_simpson to exactly match the reference implementation.
        """
        y2 = np.asarray(problem["y2"])
        dx = float(problem["dx"])
        return _cumulative_simpson(y2, dx=dx)