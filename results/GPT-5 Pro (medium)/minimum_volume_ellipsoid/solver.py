from typing import Any, Dict, Tuple
import numpy as np


class Solver:
    def _mvee(self, points: np.ndarray, tol: float = 1e-6, max_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Minimum Volume Enclosing Ellipsoid (MVEE) of a set of points.

        Args:
            points: Array of shape (n, d) with points in rows.
            tol: Tolerance for stopping criterion on Khachiyan's algorithm.
            max_iter: Maximum number of iterations.

        Returns:
            center c (d,), and shape matrix A (d,d) for ellipsoid {x | (x-c)^T A (x-c) <= 1}.
        """
        n, d = points.shape
        if n == 0 or d == 0:
            # Degenerate case: return zero-dim structures
            return np.zeros((d,)), np.eye(d)

        # Transpose to use points as columns: P is (d, n)
        P = points.T  # shape (d, n)

        # Q is (d+1, n), appending ones
        Q = np.vstack([P, np.ones((1, n))])

        # Initialize uniform weights
        u = np.full(n, 1.0 / n, dtype=float)

        m = d + 1  # dimension after lifting
        # Main loop
        for _ in range(max_iter):
            # Compute X = Q * diag(u) * Q^T without forming diag
            Qu = Q * u  # broadcast, (m, n)
            X = Qu @ Q.T  # (m, m)

            # Invert X
            try:
                invX = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                # Add tiny regularization for numerical stability
                reg = 1e-12 * np.eye(m)
                invX = np.linalg.inv(X + reg)

            # Compute M = diag(Q^T * invX * Q) efficiently:
            # M_i = q_i^T invX q_i
            V = invX @ Q  # (m, n)
            M = np.sum(Q * V, axis=0)  # (n,)

            # Find the point with maximum M
            j = int(np.argmax(M))
            Mj = float(M[j])

            # Stopping criterion
            if Mj <= m + tol:
                break

            # Khachiyan update step
            step = (Mj - m) / (m * (Mj - 1.0))
            # Ensure step is within (0,1) to avoid numerical issues
            step = max(0.0, min(1.0, step))
            # Update u
            u = (1.0 - step) * u
            u[j] += step
        else:
            # If we hit max_iter, proceed with current u (approximate)
            pass

        # Center c = P @ u
        c = P @ u  # (d,)

        # Compute covariance-like matrix S - c c^T, where S = P diag(u) P^T
        S = (P * u) @ P.T  # (d, d)
        Cov = S - np.outer(c, c)
        # Symmetrize to avoid tiny asymmetries
        Cov = 0.5 * (Cov + Cov.T)

        # Invert Cov robustly
        # If Cov is nearly singular (shouldn't be for full-dimensional point sets),
        # add a tiny ridge to ensure SPD.
        try:
            A = (1.0 / d) * np.linalg.inv(Cov)
        except np.linalg.LinAlgError:
            # Regularize
            # Scale ridge to data scale
            scale = max(1.0, float(np.linalg.norm(Cov, ord=2)))
            ridge = 1e-12 * scale
            A = (1.0 / d) * np.linalg.inv(Cov + ridge * np.eye(d))

        # Force symmetry
        A = 0.5 * (A + A.T)

        return c, A

    def solve(self, problem, **kwargs) -> Dict[str, Any]:
        """
        Solve the Minimum Volume Covering Ellipsoid problem.

        Input:
            problem: dict with key "points" -> array-like of shape (n, d)

        Output:
            dict:
              - "objective_value": float
              - "ellipsoid": {"X": (d,d) symmetric matrix, "Y": (d,) vector}
        """
        # Parse points
        points = np.array(problem["points"], dtype=float)
        if points.ndim != 2:
            # Ensure proper shape (n, d)
            points = np.atleast_2d(points)
        n, d = points.shape

        # Handle trivial cases
        if n == 0 or d == 0:
            X = np.eye(d)
            Y = np.zeros(d)
            obj = float(-np.log(np.linalg.det(X)))
            return {"objective_value": obj, "ellipsoid": {"X": X, "Y": Y}}

        # Compute MVEE center and shape
        c, A = self._mvee(points, tol=1e-6)

        # Compute X as symmetric square root of A, and Y = -X c
        # Eigen decomposition for symmetric PSD A
        w, V = np.linalg.eigh(A)
        # Clip tiny negative eigenvalues due to numerical errors
        w = np.maximum(w, 0.0)
        sqrt_w = np.sqrt(w)
        X = (V * sqrt_w) @ V.T  # symmetric sqrt
        # Symmetrize to clean numeric noise
        X = 0.5 * (X + X.T)

        Y = -X @ c

        # Ensure feasibility strictly by scaling down if needed
        # r_max = max_i ||X a_i + Y||
        XA_plus_Y = (X @ points.T).T + Y  # shape (n, d)
        norms = np.linalg.norm(XA_plus_Y, axis=1)
        r_max = float(np.max(norms)) if norms.size > 0 else 0.0

        if not np.isfinite(r_max):
            # Fallback: set identity
            X = np.eye(d)
            Y = np.zeros(d)
        else:
            # If slightly > 1, scale X and Y down to enforce feasibility
            if r_max > 1.0:
                s = 1.0 / (r_max + 1e-12)
                # Scale
                X *= s
                Y *= s

        # Objective value as per requirement: -log det X
        # Using the same computation as validator to avoid discrepancies
        try:
            detX = np.linalg.det(X)
        except np.linalg.LinAlgError:
            # Should not happen for symmetric PSD X; fallback
            detX = float(np.prod(np.linalg.eigvalsh(X)))
        # Guard against nonpositive determinant due to numerical issues
        if detX <= 0.0 or not np.isfinite(detX):
            # Add tiny ridge to X to make it SPD
            eps = 1e-12
            X = 0.5 * (X + X.T) + eps * np.eye(d)
            detX = np.linalg.det(X)

        objective_value = float(-np.log(detX))

        # Final sanity: ensure symmetry and feasibility
        X = 0.5 * (X + X.T)
        # Final small projection to PSD if needed
        evals, evecs = np.linalg.eigh(X)
        evals = np.maximum(evals, 0.0)
        X = (evecs * evals) @ evecs.T
        X = 0.5 * (X + X.T)

        # Recompute Y to stay consistent after projection
        # Note: Projecting X might slightly change feasibility; rescale again if needed
        # Keep the same center c to maintain structure Y = -X c
        Y = -X @ c

        XA_plus_Y = (X @ points.T).T + Y
        norms = np.linalg.norm(XA_plus_Y, axis=1)
        r_max = float(np.max(norms)) if norms.size > 0 else 0.0
        if r_max > 1.0 + 1e-8:
            s = 1.0 / (r_max + 1e-12)
            X *= s
            Y *= s
            # Update objective accordingly
            try:
                detX = np.linalg.det(X)
            except np.linalg.LinAlgError:
                detX = float(np.prod(np.linalg.eigvalsh(X)))
            if detX <= 0.0 or not np.isfinite(detX):
                # minimal fix
                X = 0.5 * (X + X.T) + 1e-12 * np.eye(d)
                detX = np.linalg.det(X)
            objective_value = float(-np.log(detX))
        else:
            # Recompute objective in case X changed
            try:
                detX = np.linalg.det(X)
            except np.linalg.LinAlgError:
                detX = float(np.prod(np.linalg.eigvalsh(X)))
            if detX <= 0.0 or not np.isfinite(detX):
                X = 0.5 * (X + X.T) + 1e-12 * np.eye(d)
                detX = np.linalg.det(X)
            objective_value = float(-np.log(detX))

        return {"objective_value": objective_value, "ellipsoid": {"X": X, "Y": Y}}