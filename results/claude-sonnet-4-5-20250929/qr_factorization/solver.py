import numpy as np
from scipy import linalg

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the QR factorization problem by computing the QR factorization of matrix A.
        """
        A = problem["matrix"]
        Q, R = linalg.qr(A, mode="economic")
        solution = {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
        return solution