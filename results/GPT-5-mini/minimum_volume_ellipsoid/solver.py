from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Compute the Minimum Volume Enclosing Ellipsoid (MVEE) using Khachiyan's algorithm
        (augmented formulation). The ellipsoid is returned in the form:

            || X * a_i + Y ||_2 <= 1  for all input points a_i,

        where X is symmetric positive definite and Y is a vector.

        Returns:
            {
                "objective_value": float,   # -log det(X)
                "ellipsoid": {"X": [[...]], "Y": [...]}
            }
        """
        try:
            pts = np.array(problem.get("points", []), dtype=float)

            # Normalize shape
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.size == 0:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": [], "Y": []},
                }

            n, d = pts.shape

            # Single point: trivial ellipsoid (identity X, center at the point)
            if n == 1:
                c = pts[0]
                X_id = np.eye(d)
                Y_vec = -X_id.dot(c)
                sign, ld = np.linalg.slogdet(X_id)
                if sign <= 0 or not np.isfinite(ld):
                    return {
                        "objective_value": float("inf"),
                        "ellipsoid": {
                            "X": (np.nan * np.ones((d, d))).tolist(),
                            "Y": (np.nan * np.ones((d,))).tolist(),
                        },
                    }
                objective = -float(ld)
                return {"objective_value": objective, "ellipsoid": {"X": X_id.tolist(), "Y": Y_vec.tolist()}}

            # Algorithm hyperparameters (can be overridden)
            tol = float(kwargs.get("tol", 1e-8))
            max_iter = int(kwargs.get("max_iter", 1000))
            reg = float(kwargs.get("reg", 1e-12))
            tol_feas = float(kwargs.get("tol_feas", 1e-9))

            # Augmented matrix Q = [points; 1]
            Q = np.vstack((pts.T, np.ones((1, n))))  # shape (d+1, n)
            m = d + 1

            # Initialize weights
            u = np.ones(n, dtype=float) / float(n)

            # Khachiyan iterations (augmented variant)
            for _ in range(max_iter):
                # Xk = Q * diag(u) * Q^T
                Xk = Q @ (Q.T * u)

                # Robust inverse (fallback to pseudo-inverse)
                try:
                    invXk = np.linalg.inv(Xk)
                except np.linalg.LinAlgError:
                    invXk = np.linalg.pinv(Xk)

                # Compute M_i = q_i^T invXk q_i for all i
                tmp = invXk @ Q  # (d+1, n)
                M = np.einsum("ij,ij->j", Q, tmp)  # (n,)

                j = int(np.argmax(M))
                maxM = float(M[j])

                # Convergence check
                if maxM <= m * (1.0 + tol):
                    break

                denom = (maxM - 1.0)
                if denom <= 0:
                    break

                step = (maxM - m) / (m * denom)
                if step <= 0:
                    break

                new_u = (1.0 - step) * u
                new_u[j] += step

                # Small update check
                if np.linalg.norm(new_u - u) <= tol * max(1.0, np.linalg.norm(u)):
                    u = new_u
                    break
                u = new_u

            # Compute center c and scatter matrix S
            c = pts.T.dot(u)  # (d,)

            # S = sum_i u_i (p_i - c)(p_i - c)^T
            P_centered = pts - c  # (n, d)
            S = (P_centered.T * u) @ P_centered  # (d, d)

            # Stabilize S via eigen-decomposition (handles near-singular cases)
            try:
                eigvals, eigvecs = np.linalg.eigh(S)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eig(S)
                eigvals = eigvals.real
                eigvecs = eigvecs.real

            # Clip eigenvalues to ensure invertibility
            abs_max = np.max(np.abs(eigvals)) if eigvals.size > 0 else 0.0
            eig_eps = max(reg, 1e-16 * max(1.0, abs_max))
            eigvals_clipped = np.clip(eigvals, eig_eps, None)
            inv_eigs = 1.0 / eigvals_clipped

            invS = (eigvecs * inv_eigs) @ eigvecs.T

            # A = inv(S) / d  (ellipsoid quadratic form)
            A = invS / float(d)
            A = 0.5 * (A + A.T)

            # Compute symmetric PD square-root X such that X @ X = A
            try:
                a_eigs, a_vecs = np.linalg.eigh(A)
            except np.linalg.LinAlgError:
                a_eigs, a_vecs = np.linalg.eig(A)
                a_eigs = a_eigs.real
                a_vecs = a_vecs.real

            a_eps = max(reg, 1e-20)
            a_eigs_clipped = np.clip(a_eigs, a_eps, None)
            sqrt_eigs = np.sqrt(a_eigs_clipped)
            X_mat = (a_vecs * sqrt_eigs) @ a_vecs.T
            X_mat = 0.5 * (X_mat + X_mat.T)

            # Y such that X * v + Y = X (v - c)
            Y_vec = -X_mat.dot(c)

            # Feasibility correction: scale down if numerical violations occur
            vals = X_mat.dot(pts.T) + Y_vec[:, None]
            norms = np.linalg.norm(vals, axis=0)
            maxnorm = float(np.max(norms))
            if maxnorm > 1.0 + tol_feas:
                scale = maxnorm
                X_mat = X_mat / scale
                Y_vec = Y_vec / scale

            # Ensure numerical positive-definiteness
            try:
                evals = np.linalg.eigvalsh(X_mat)
                if np.min(evals) <= 0:
                    evals2, vecs2 = np.linalg.eigh(0.5 * (X_mat + X_mat.T))
                    evals2_clipped = np.clip(evals2, a_eps, None)
                    X_mat = (vecs2 * evals2_clipped) @ vecs2.T
                    X_mat = 0.5 * (X_mat + X_mat.T)
            except np.linalg.LinAlgError:
                X_mat = X_mat + reg * np.eye(d)

            # Objective: -log det(X)
            sign, ld = np.linalg.slogdet(X_mat)
            if sign <= 0 or not np.isfinite(ld):
                nan_mat = (np.nan * np.ones((d, d))).tolist()
                nan_vec = (np.nan * np.ones((d,))).tolist()
                return {"objective_value": float("inf"), "ellipsoid": {"X": nan_mat, "Y": nan_vec}}

            objective = -float(ld)

            return {
                "objective_value": objective,
                "ellipsoid": {"X": X_mat.tolist(), "Y": Y_vec.tolist()},
            }

        except Exception:
            # On unexpected failure return NaN structure similar to reference fallback
            d = 0
            try:
                pts0 = np.array(problem.get("points", []), dtype=float)
                if pts0.ndim == 1:
                    pts0 = pts0.reshape(1, -1)
                if pts0.size != 0:
                    d = int(pts0.shape[1])
            except Exception:
                d = 0
            nan_mat = (np.nan * np.ones((d, d))).tolist()
            nan_vec = (np.nan * np.ones((d,))).tolist()
            return {"objective_value": float("inf"), "ellipsoid": {"X": nan_mat, "Y": nan_vec}}