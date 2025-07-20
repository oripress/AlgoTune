import numpy as np
from scipy.linalg import cholesky

class Solver:
    def solve(self, problem, **kwargs):
        """Compute the Cholesky factorization of a symmetric positive definite matrix."""
        A = np.array(problem["matrix"], dtype=np.float64)
        L = cholesky(A, lower=True)
        solution = {"Cholesky": {"L": L}}
        return solution