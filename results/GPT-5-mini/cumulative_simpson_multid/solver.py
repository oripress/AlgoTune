import numpy as np
from typing import Any
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Compute the cumulative integral along the last axis using SciPy's
        cumulative_simpson to exactly match the reference implementation.
        """
        y = np.asarray(problem["y2"])
        dx = float(problem["dx"])
        return cumulative_simpson(y, dx=dx, axis=-1)