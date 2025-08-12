from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Compute the projection of a symmetric matrix A onto the PSD cone.

        Args:
            problem: A dictionary with key "A" containing an n x n symmetric matrix.

        Returns:
            A dictionary with key "X" containing the PSD projection of A.
        """
        A = np.asarray(problem["A"], dtype=float, order="F")
        n = A.shape[0]

        # Small-size fast paths
        if n == 1:
            a = A[0, 0]
            return {"X": np.array([[a if a > 0.0 else 0.0]], dtype=A.dtype)}

        if n == 2:
            a = A[0, 0]
            b = A[0, 1]
            c = A[1, 1]
            # Diagonal case
            if b == 0.0:
                return {"X": np.array([[a if a > 0.0 else 0.0, 0.0],
                                       [0.0, c if c > 0.0 else 0.0]], dtype=A.dtype)}
            # Eigenvalues
            t = a + c
            d = np.hypot(a - c, 2.0 * b)
            lam1 = 0.5 * (t - d)
            lam2 = 0.5 * (t + d)
            if lam2 <= 0.0:
                return {"X": np.zeros((2, 2), dtype=A.dtype)}
            if lam1 >= 0.0:
                return {"X": A}

            # One positive, one non-positive: X = lam2 * u u^T, with u eigenvector of lam2
            # Use u = [b, lam2 - a]; fall back if degenerate
            u0 = b
            u1 = lam2 - a
            nrm2 = u0 * u0 + u1 * u1
            if nrm2 == 0.0:
                # Degenerate (nearly diagonal): choose axis based on which equals lam2
                if abs(lam2 - a) <= abs(lam2 - c):
                    u0, u1 = 1.0, 0.0
                else:
                    u0, u1 = 0.0, 1.0
                nrm2 = 1.0
            scale = lam2 / nrm2
            x00 = scale * (u0 * u0)
            x01 = scale * (u0 * u1)
            x11 = scale * (u1 * u1)
            X2 = np.array([[x00, x01], [x01, x11]], dtype=A.dtype)
            return {"X": X2}

        # Eigendecomposition for symmetric matrices (ascending eigenvalues)
        w, Q = np.linalg.eigh(A, UPLO="L")

        wmin = w[0]
        wmax = w[-1]

        # All non-positive: projection is zero
        if wmax <= 0.0:
            return {"X": np.zeros((n, n), dtype=A.dtype)}

        # All non-negative: A is already PSD
        if wmin >= 0.0:
            return {"X": A}

        # Mixed signs: use only the positive-eigenvalue subspace
        start = np.searchsorted(w, 0.0, side="right")
        Qp = Q[:, start:]
        wp = w[start:]
        X = (Qp * wp) @ Qp.T
        return {"X": X}