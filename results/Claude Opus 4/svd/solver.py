import numpy as np
from numba import jit
from typing import Any

class Solver:
    def __init__(self):
        # Pre-compile the JIT function
        self._svd_wrapper = self._create_svd_wrapper()
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _create_svd_wrapper():
        def svd_wrapper(A):
            return np.linalg.svd(A, full_matrices=False)
        return svd_wrapper
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the SVD problem by computing the singular value decomposition of matrix A.
        """
        A = np.array(problem["matrix"], dtype=np.float64)
        
        # Use JIT-compiled SVD
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T
        
        solution = {"U": U.tolist(), "S": s.tolist(), "V": V.tolist()}
        return solution