import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        """Compute the matrix exponential directly."""
        # Convert to numpy array for better performance
        A = np.array(problem["matrix"])
        expA = expm(A)
        return {"exponential": expA.tolist()}