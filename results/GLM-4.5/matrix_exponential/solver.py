import numpy as np
from numba import jit
from scipy.linalg import expm

@jit(nopython=True)
def matrix_exp_taylor(A, max_terms=20):
    """Compute matrix exponential using Taylor series expansion with Numba JIT."""
    n = A.shape[0]
    expA = np.eye(n, dtype=A.dtype)
    term = np.eye(n, dtype=A.dtype)
    
    for k in range(1, max_terms):
        term = term @ A / k
        expA = expA + term
        
    return expA

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        A = np.array(problem["matrix"])
        
        # Use Numba-accelerated Taylor series for small matrices
        if A.shape[0] <= 5:
            expA = matrix_exp_taylor(A)
        else:
            expA = expm(A)
        
        return {"exponential": expA.tolist()}