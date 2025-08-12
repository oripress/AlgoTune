import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs) -> tuple:
        """Solve generalized eigenvalue problem A·x = λB·x"""
        A, B = problem
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64, order='F')
        if not isinstance(B, np.ndarray):
            B = np.asarray(B, dtype=np.float64, order='F')
        eigenvalues, eigenvectors = eigh(A, B, check_finite=False)
        return (eigenvalues[::-1].tolist(), eigenvectors[:, ::-1].T.tolist())