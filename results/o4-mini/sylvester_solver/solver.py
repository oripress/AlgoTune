from typing import Any, Dict
import numpy as np
from scipy.linalg import schur
from numba import njit

@njit(fastmath=True)
def _stsyl(T1, T2, D):
    # Solve T1*Y + Y*T2 = D for Y (both T1 and T2 are upper-triangular)
    N = T1.shape[0]
    M = T2.shape[0]
    Y = np.empty_like(D)
    for ii in range(N-1, -1, -1):
        for jj in range(M):
            s = D[ii, jj]
            # subtract contributions from already computed entries
            for k in range(ii+1, N):
                s -= T1[ii, k] * Y[k, jj]
            for k in range(jj):
                s -= Y[ii, k] * T2[k, jj]
            # solve diagonal element
            Y[ii, jj] = s / (T1[ii, ii] + T2[jj, jj])
    return Y

class Solver:
    def __init__(self):
        # Warm-up Numba compilation (won't count toward solve runtime)
        one = np.array([[1+0j]], dtype=np.complex128)
        _stsyl(one, one, one)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]

        # Compute Schur decompositions (in-place, no checks)
        T1, U = schur(A, output='complex', overwrite_a=True, check_finite=False)
        T2, V = schur(B, output='complex', overwrite_a=True, check_finite=False)

        # Transform RHS into Schur basis: D = U^H * Q * V
        D = U.conj().T.dot(Q).dot(V)

        # Solve triangular Sylvester via Numba-compiled routine
        Y = _stsyl(T1, T2, D)

        # Back-transform solution: X = U * Y * V^H
        X = U.dot(Y).dot(V.conj().T)
        return {"X": X}