import numpy as np
from typing import Any
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Compute the cumulative integral of a 1‑D array using Simpson's rule.
        This implementation directly uses SciPy's `cumulative_simpson`
        to ensure exact matching of the reference implementation.
        """
        y = np.asarray(problem["y"], dtype=float)
        dx = float(problem["dx"])
        # SciPy's cumulative_simpson returns the cumulative integral
        # with the same behavior as the reference implementation.
        return cumulative_simpson(y, dx=dx)