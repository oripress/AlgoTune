import numpy as np
from scipy.linalg.lapack import dpotrf

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the Cholesky factorization problem using direct LAPACK dpotrf call.
        
        :param problem: A dictionary representing the Cholesky factorization problem.
        :return: A dictionary with key "Cholesky" containing a dictionary with key:
                 "L": A list of lists representing the lower triangular matrix L.
        """
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64, order='C')
        elif A.dtype != np.float64 or not A.flags['C_CONTIGUOUS']:
            A = np.ascontiguousarray(A, dtype=np.float64)
        
        # Use LAPACK dpotrf directly for speed with clean=True
        L, info = dpotrf(A, lower=True, clean=True, overwrite_a=True)
        
        solution = {"Cholesky": {"L": L}}
        return solution