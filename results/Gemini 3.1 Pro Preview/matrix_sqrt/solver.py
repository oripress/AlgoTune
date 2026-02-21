import numpy as np
import scipy.linalg
from numba import njit

@njit(fastmath=True)
def fast_sqrtm(A):
    n = A.shape[0]
    is_herm = True
    for i in range(n):
        for j in range(n):
            if A[i, j] != np.conj(A[j, i]):
                is_herm = False
                break
        if not is_herm:
            break
            
    if is_herm:
        evals, evecs = np.linalg.eigh(A)
        sqrt_evals = np.sqrt(evals + 0j)
        return (evecs * sqrt_evals) @ np.conj(evecs.T)
    else:
        evals, evecs = np.linalg.eig(A)
        sqrt_evals = np.sqrt(evals)
        return np.linalg.solve(evecs.T, (evecs * sqrt_evals).T).T

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        A = np.asarray(problem["matrix"], dtype=complex)
        try:
            X = fast_sqrtm(A)
            return {"sqrtm": {"X": X.tolist()}}
        except Exception:
            try:
                X, _ = scipy.linalg.sqrtm(A, disp=False)
                return {"sqrtm": {"X": X.tolist()}}
            except Exception:
                return {"sqrtm": {"X": []}}