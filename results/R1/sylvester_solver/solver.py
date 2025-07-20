import numpy as np
from scipy.linalg import schur
import numba

@numba.jit(nopython=True, cache=True)
def solve_triangular(R, S, C):
    n = R.shape[0]
    m = S.shape[0]
    Y = np.zeros((n, m), dtype=np.complex128)
    for j in range(m):
        for i in range(n-1, -1, -1):
            s1 = 0.0j
            for k in range(i+1, n):
                s1 += R[i, k] * Y[k, j]
            s2 = 0.0j
            for k in range(0, j):
                s2 += Y[i, k] * S[k, j]
            denom = R[i, i] + S[j, j]
            Y[i, j] = (C[i, j] - s1 - s2) / denom
    return Y

# Precompile the function
try:
    dummy_R = np.zeros((1,1), dtype=np.complex128)
    dummy_S = np.zeros((1,1), dtype=np.complex128)
    dummy_C = np.zeros((1,1), dtype=np.complex128)
    solve_triangular(dummy_R, dummy_S, dummy_C)
except:
    pass

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        
        # Compute Schur decompositions
        R, U = schur(A, output='complex')
        S, V = schur(B, output='complex')
        
        # Transform equation: U^H Q V = R Y + Y S
        C = U.conj().T @ Q @ V
        
        # Solve triangular system
        Y = solve_triangular(R, S, C)
        
        # Transform back to original basis
        X = U @ Y @ V.conj().T
        return {"X": X}