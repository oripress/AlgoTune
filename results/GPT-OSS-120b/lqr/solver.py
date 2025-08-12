import numpy as np
from typing import Any, Dict
import numba

import numpy as np
from typing import Any, Dict

def _riccati_backward(A, B, Q, R, P, T):
    n, m = B.shape
    K = np.empty((T, m, n), dtype=A.dtype)
    S = P.copy()
    for t in range(T - 1, -1, -1):
        M1 = R + B.T @ S @ B
        M2 = B.T @ S @ A
        # Solve (R + BᵀSB) Kᵀ = BᵀSA  →  K = M1⁻¹ M2
        K[t] = np.linalg.solve(M1, M2)
        Acl = A - B @ K[t]
        S = Q + K[t].T @ R @ K[t] + Acl.T @ S @ Acl
        S = (S + S.T) * 0.5  # enforce symmetry
    return K

def _forward_simulation(A, B, K, x0, T):
    n, m = B.shape
    U = np.empty((T, m), dtype=A.dtype)
    x = x0.copy()
    for t in range(T):
        u = -K[t] @ x
        U[t] = u.ravel()
        x = A @ x + B @ u
    return U
class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the discrete‑time LQR problem using Riccati recursion and forward simulation.
        Returns a dictionary with key "U" containing the optimal control sequence.
        """
        # Convert inputs to NumPy arrays (float64) and proper shapes
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        Q = np.asarray(problem["Q"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=np.float64).reshape(-1, 1)

        # Backward Riccati recursion to obtain feedback gains
        K = _riccati_backward(A, B, Q, R, P, T)

        # Forward simulation to compute optimal control sequence
        U = _forward_simulation(A, B, K, x0, T)

        return {"U": U.tolist()}