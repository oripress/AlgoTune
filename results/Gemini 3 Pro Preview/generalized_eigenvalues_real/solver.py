import scipy.linalg
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        # scipy.linalg.eigh can solve the generalized eigenvalue problem directly.
        # type=1 solves A*x = lambda*B*x
        # eigvals_only=True returns only the eigenvalues, which is faster.
        eigenvalues = scipy.linalg.eigh(A, B, eigvals_only=True, type=1)
        
        # The eigenvalues are returned in ascending order, so we reverse them.
        return eigenvalues[::-1].tolist()