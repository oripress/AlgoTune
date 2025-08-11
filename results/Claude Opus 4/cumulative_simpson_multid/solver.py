import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array using Simpson's rule.
        """
        y2 = problem["y2"]
        dx = problem["dx"]
        
        # Use scipy's cumulative_simpson as a starting point
        result = cumulative_simpson(y2, dx=dx)
        return result