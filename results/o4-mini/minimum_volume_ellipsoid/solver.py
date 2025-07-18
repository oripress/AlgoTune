import numpy as np
from scipy.linalg import cholesky, solve_triangular
class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Minimum Volume Covering Ellipsoid (MVEE) using Khachiyan's algorithm.

        Args:
            problem: dict with key "points" for the list of points (n x d).

        Returns:
            dict with:
              - "objective_value": float
              - "ellipsoid": { "X": list of lists, "Y": list }
        """
        points = np.array(problem["points"], dtype=float)
        n, d = points.shape

        # Build data matrix Q with shape (n, d+1)
        Q = np.hstack((points, np.ones((n, 1), dtype=float)))
        Q_T = Q.T
        solve_fn = np.linalg.solve
        u = np.ones(n, dtype=float) / n

        tol = 1e-6
        max_iter = 500
        for _ in range(max_iter):
            # Weighted data and moment matrix
            Xu = Q * u[:, None]          # shape (n, d+1)
            Xmat = Q_T @ Xu              # shape (d+1, d+1)

            # Solve linear systems to get invXQ (shape (d+1, n))
            try:
                invXQ = solve_fn(Xmat, Q_T)
            except np.linalg.LinAlgError:
                break

            # Mahalanobis distances: M[i] = Q[i] dot invXQ[:,i]
            M = (Q * invXQ.T).sum(axis=1)  # shape (n,)

            j = np.argmax(M)
            maxM = M[j]
            # Step size
            step = (maxM - d - 1.0) / ((d + 1.0) * (maxM - 1.0))
            if step <= tol:
                break
            # Update weights
            u = (1.0 - step) * u
            u[j] += step
        # Compute center of ellipsoid
        center = points.T @ u

        # Compute covariance-like matrix
        cov = (points.T * u) @ points - np.outer(center, center)
        cov = (cov + cov.T) / 2.0  # symmetrize

        # Shape matrix A of the ellipsoid: (x - center)^T A (x - center) <= 1
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        A = inv_cov / float(d)
        A = (A + A.T) / 2.0

        # Compute mapping matrix X = sqrtm(A)
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.clip(eigvals, 0.0, None)
        sqrt_eigs = np.sqrt(eigvals)
        X_mat = (eigvecs * sqrt_eigs) @ eigvecs.T

        # Compute Y so that ||X_mat * x + Y||_2 <= 1 defines the ellipsoid
        Y = -X_mat @ center

        # Ensure all points are inside ellipsoid; adjust scale if necessary
        vals = X_mat @ points.T + Y[:, None]
        norms = np.linalg.norm(vals, axis=0)
        max_norm = norms.max() if norms.size > 0 else 1.0
        if max_norm > 1.0:
            scale = max_norm
            X_mat /= scale
            Y /= scale

        # Objective value
        detX = np.linalg.det(X_mat)
        if detX <= 0:
            obj = float('inf')
        else:
            obj = -np.log(detX)

        return {
            "objective_value": float(obj),
            "ellipsoid": {
                "X": X_mat.tolist(),
                "Y": Y.tolist()
            }
        }