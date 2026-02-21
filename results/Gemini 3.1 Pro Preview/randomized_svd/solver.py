from typing import Any
import numpy as np
import scipy.linalg as sla

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float32)
        elif A.dtype != np.float32:
            A = A.astype(np.float32, copy=False)
            
        k = problem["n_components"]
        n, m = A.shape
        
        if k > min(n, m) // 2:
            U, S, Vt = sla.svd(A, full_matrices=False, overwrite_a=True, check_finite=False)
            return {"U": U[:, :k], "S": S[:k], "V": Vt[:k, :].T}
            
        Y = A[:, :k].copy()
        Q, _ = sla.qr(Y, mode='economic', overwrite_a=True, check_finite=False)
            
        B = np.dot(Q.T, A)
        U_tilde, S, Vt = sla.svd(B, full_matrices=False, overwrite_a=True, check_finite=False)
        U = np.dot(Q, U_tilde)
        
        return {"U": U, "S": S, "V": Vt.T}