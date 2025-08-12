import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> Dict[str, Any]:
        """
        Solve the Minimum Volume Covering Ellipsoid problem using the
        Khachiyan algorithm (an efficient iterative method).

        Parameters
        ----------
        problem : dict
            Dictionary with key "points" containing a list of n points,
            each a list of d floats.

        Returns
        -------
        dict
            {
                "objective_value": float,
                "ellipsoid": {
                    "X": np.ndarray of shape (d, d),
                    "Y": np.ndarray of shape (d,)
                }
            }
        """
        points = np.asarray(problem["points"], dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("Points must be a 2‑dimensional array.")
        n, d = points.shape

        # ---------- Khachiyan algorithm ----------
        # Initialise uniform weights
        u = np.full(n, 1.0 / n)

        # Convergence parameters
        tol = 1e-7
        max_iter = 1000

        # Pre‑compute transpose for speed
        P = points.T  # d x n

        for _ in range(max_iter):
            # Current center
            c = P @ u  # d

            # Centered points
            diff = P - c[:, None]  # d x n

            # Weighted covariance matrix
            S = diff @ (u * diff.T)  # d x d  (equivalent to diff @ np.diag(u) @ diff.T)

            # Solve S^{-1} * diff efficiently
            try:
                invS_diff = np.linalg.solve(S, diff)  # d x n
            except np.linalg.LinAlgError:
                # Numerical issue – fall back to pseudo‑inverse
                invS_diff = np.linalg.pinv(S) @ diff

            # Mahalanobis distances squared
            M = np.einsum("ij,ij->j", diff, invS_diff)  # length n

            # Find the point with maximum violation
            j = np.argmax(M)
            max_M = M[j]

            # Check convergence (max_M should be <= d)
            if max_M <= d + tol:
                break

            # Compute step size
            step = (max_M - d) / ((d + 1) * max_M)

            # Update weights
            u = (1 - step) * u
            u[j] += step

        # Final center and shape matrix
        c = P @ u
        diff = P - c[:, None]
        S = diff @ (u * diff.T)

        # Shape matrix A = (1/d) * S^{-1}
        A = np.linalg.inv(S) / d

        # Cholesky factor L such that A = L @ L.T
        L = np.linalg.cholesky(A)

        # According to formulation: X v + Y = L (v - c)
        X_mat = L
        Y_vec = -L @ c

        # Objective value: -log(det(X)) = -log(det(L))
        obj_val = -np.log(np.linalg.det(X_mat))

        return {
            "objective_value": float(obj_val),
            "ellipsoid": {
                "X": X_mat,
                "Y": Y_vec,
            },
        }