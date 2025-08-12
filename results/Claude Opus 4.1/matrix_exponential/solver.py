import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict, **kwargs):
        """Compute the matrix exponential exp(A)."""
        A = np.array(problem["matrix"])
        expA = expm(A)
        return {"exponential": expA.tolist()}