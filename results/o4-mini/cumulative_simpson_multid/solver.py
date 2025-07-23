import numpy as np
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the cumulative integral along the last axis of the multi-dimensional
        array using Simpson's rule via SciPy's cumulative_simpson.
        """
        y = problem["y2"]
        dx = problem["dx"]
        # Convert to NumPy array with appropriate dtype
        y = np.asarray(y, dtype=np.float64)
        # Use SciPy's cumulative_simpson for correctness
        result = cumulative_simpson(y, dx=dx, axis=-1)
        return result