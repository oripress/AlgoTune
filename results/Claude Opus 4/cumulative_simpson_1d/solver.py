import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        """
        y = problem["y"]
        dx = problem["dx"]
        
        # Reference implementation for now
        from scipy.integrate import cumulative_simpson
        return cumulative_simpson(y, dx=dx)