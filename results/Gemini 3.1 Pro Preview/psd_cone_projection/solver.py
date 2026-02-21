import numpy as np
from typing import Any
import numba as nb

@nb.njit(fastmath=True)
def project_psd(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    n = eigvals.shape[0]
    for i in range(n):
        if eigvals[i] > 0:
            val = np.sqrt(eigvals[i])
            for j in range(n):
                eigvecs[j, i] *= val
        else:
            for j in range(n):
                eigvecs[j, i] = 0.0
                
    if n < 50:
        out = np.empty((n, n), dtype=A.dtype)
        for i in range(n):
            for j in range(i, n):
                sum_val = 0.0
                for k in range(n):
                    sum_val += eigvecs[i, k] * eigvecs[j, k]
                out[i, j] = sum_val
                out[j, i] = sum_val
        return out
    else:
        return np.dot(eigvecs, eigvecs.T)
class Solver:
    def __init__(self):
        dummy = np.eye(2, dtype=np.float64)
        project_psd(dummy)

    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, Any]:
        A = np.asarray(problem["A"])
        X = project_psd(A)
        return {"X": X}