import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array using Simpson's rule.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        y2 = np.asarray(problem["y2"], dtype=np.float64)
        dx = problem["dx"]
        result = cumulative_simpson(y2, dx=dx)
        return result