import numpy as np
import numba as nb

@nb.njit
def compute_frobenius_norm_squared(B):
    """Compute Frobenius norm squared using numba for speed."""
    n = B.shape[0]
    sum_sq = 0.0
    for i in range(n):
        for j in range(n):
            sum_sq += B[i, j] * B[i, j]
    return sum_sq

class Solver:
    def __init__(self):
        # Warm up numba
        dummy = np.array([[1.0, 2.0], [3.0, 4.0]])
        compute_frobenius_norm_squared(dummy)
    
    def solve(self, problem):
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        """
        A, B = problem
        
        # Scale matrices for better numerical stability
        # Use numba-compiled Frobenius norm
        scale_B = np.sqrt(compute_frobenius_norm_squared(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B
        
        # Solve scaled problem using numpy's eig
        eigenvalues = np.linalg.eigvals(A_scaled @ np.linalg.inv(B_scaled))
        
        # Sort eigenvalues: descending order by real part, then by imaginary part
        idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        return eigenvalues[idx].tolist()