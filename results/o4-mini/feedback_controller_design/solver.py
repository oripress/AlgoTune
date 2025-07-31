import numpy as np
from scipy.linalg import solve_discrete_are

# Pre-bind common functions for speed
_solve_dare = solve_discrete_are
_arr = np.asarray
_solve = np.linalg.solve
_eye = np.eye

class Solver:
    def solve(self, problem, **kwargs):
        try:
            # Local references
            arr = _arr; eye = _eye
            dare = _solve_dare; lin_solve = _solve

            # Load system matrices
            A = arr(problem["A"])
            B = arr(problem["B"])
            n, m = A.shape[0], B.shape[1]

            # Solve discrete-time Riccati equation
            P = dare(A, B, eye(n), eye(m))

            # Compute gain K = -(R + B'PB)^{-1} B'PA
            BtP = B.T @ P
            G = eye(m) + BtP @ B
            H = BtP @ A
            K = -lin_solve(G, H)

            # Symmetrize P and return arrays directly
            P = 0.5 * (P + P.T)
            return {"is_stabilizable": True, "K": K, "P": P}
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}