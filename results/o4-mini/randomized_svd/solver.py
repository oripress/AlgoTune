import numpy as np
from numpy.linalg import qr, svd
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Load matrix as float32 for faster computation
        A = np.array(problem["matrix"], dtype=np.float32)
        n, m = A.shape
        k = problem["n_components"]

        # Choose smaller side for sketch
        if n >= m:
            M = A.T  # shape (m, n)
            transpose = True
        else:
            M = A    # shape (n, m)
            transpose = False

        # Random test matrix (no oversampling)
        np.random.seed(42)
        Omega = np.random.randn(M.shape[1], k).astype(np.float32)

        # Sketch and orthonormalize
        Y = M @ Omega
        Q, _ = qr(Y, mode="reduced")

        # Project and compute small SVD
        B = Q.T @ M
        Ub, s, Vt = svd(B, full_matrices=False)

        # Build approximate U, V
        if transpose:
            # From SVD of A.T: A ≈ V S U^T => U = V_m, V = U_m
            U = Vt.T  # shape (n, k)
            V = Q @ Ub  # shape (m, k)
        else:
            U = Q @ Ub  # shape (n, k)
            V = Vt.T  # shape (m, k)

        # Cast back to float64 for output consistency
        return {"U": U.astype(np.float64),
                "S": s[:k].astype(np.float64),
                "V": V.astype(np.float64)}