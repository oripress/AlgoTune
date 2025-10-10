import numpy as np
import scipy.linalg
from typing import Any
from numba import jit

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the SVD problem using optimized scipy implementation.
        """
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        
        # Use scipy's SVD with gesdd driver (divide and conquer, generally faster)
        U, s, Vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesdd', check_finite=False)
        V = Vh.T
        
        solution = {"U": U, "S": s, "V": V}
        return solution