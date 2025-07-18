import numpy as np
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array
        using Simpson's rule by leveraging SciPy's optimized implementation.
        """
        y2 = np.asarray(problem["y2"], dtype=float)
        dx = float(problem["dx"])
        return cumulative_simpson(y2, dx=dx)