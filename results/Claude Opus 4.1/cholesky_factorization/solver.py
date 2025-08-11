import numpy as np
from scipy.linalg import cholesky

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Cholesky factorization problem by computing the Cholesky decomposition of matrix A.
        """
        A = problem["matrix"]
        L = cholesky(A, lower=True)
        solution = {"Cholesky": {"L": L}}
        return solution