import numpy as np
from numba import jit
import scipy.linalg

@jit(nopython=True)
def _lu_decomp_numba(A):
    """Perform LU decomposition with partial pivoting using Numba"""
    n = A.shape[0]
    # Create copies
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    
    for k in range(n-1):
        # Find pivot
        max_idx = k
        max_val = abs(U[k, k])
        for i in range(k+1, n):
            if abs(U[i, k]) > max_val:
                max_val = abs(U[i, k])
                max_idx = i
        
        # Swap rows if needed
        if max_idx != k:
            # Swap in U
            for j in range(n):
                U[k, j], U[max_idx, j] = U[max_idx, j], U[k, j]
            # Swap in P
            for j in range(n):
                P[k, j], P[max_idx, j] = P[max_idx, j], P[k, j]
            # Swap in L (already computed part)
            for j in range(k):
                L[k, j], L[max_idx, j] = L[max_idx, j], L[k, j]
        
        # Compute multipliers
        if U[k, k] != 0:
            for i in range(k+1, n):
                L[i, k] = U[i, k] / U[k, k]
                # Update U
                for j in range(k, n):
                    U[i, j] -= L[i, k] * U[k, j]
    
    return P, L, U

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the LU factorization problem by computing the LU factorization of matrix A."""
        A = np.array(problem["matrix"], dtype=np.float64)
        
        # For small matrices, use the optimized numba version
        if A.shape[0] <= 100:
            P, L, U = _lu_decomp_numba(A)
        else:
            # For larger matrices, fall back to scipy
            P, L, U = scipy.linalg.lu(A)
            
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}