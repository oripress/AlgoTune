import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the LU factorization problem by computing the LU factorization of matrix A.
        """
        A = problem["matrix"]
        P, L, U = lu(A)
        solution = {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
        return solution