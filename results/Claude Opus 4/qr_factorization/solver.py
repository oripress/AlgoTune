import numpy as np
from scipy.linalg import qr

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the QR factorization problem by computing the QR factorization of matrix A.
        
        :param problem: A dictionary representing the QR factorization problem.
        :return: A dictionary with key "QR" containing Q and R matrices.
        """
        A = np.array(problem["matrix"])
        Q, R = qr(A, mode="economic")
        solution = {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
        return solution