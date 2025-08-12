from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

try:
    # SciPy can provide faster/stable Cholesky solves if available
    from scipy.linalg import cho_factor, cho_solve
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

class Solver:
    def _mvee(self, P: np.ndarray, tol: float = 1e-4, max_iter: int = 10_000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Minimum Volume Enclosing Ellipsoid (MVEE) of a set of points using an optimized
        Khachiyan algorithm with rank-one updates, maintaining B = X^{-1} Q directly.

        Given points matrix P of shape (d, n) (points are columns), returns:
        - c: center vector in R^d
        - S: PSD matrix in R^{dxd} such that A = (1/d) * inv(S) defines the ellipsoid (x - c)^T A (x - c) <= 1

        Recovery:
            c = P u,    sum(u)=1, u >= 0
            S = P diag(u) P^T - c c^T
        """
        d, n = P.shape
        if n == 0:
            return np.zeros(d), np.eye(d)

        # Augmented data Q = [P; 1], size (d+1) x n
        Q = np.vstack((P, np.ones((1, n), dtype=P.dtype)))
        QT = Q.T  # ensure contiguous for faster matmul
        d1 = d + 1

        # Initialize uniform weights
        u = np.full(n, 1.0 / n, dtype=P.dtype)

        # X = Q diag(u) Q^T
        X = (Q * u) @ Q.T

        # Inverse of X (implicitly) by solving against Q to form B = X^{-1} Q
        I = np.eye(d1, dtype=P.dtype)
        try:
            if _HAVE_SCIPY:
                cfac = cho_factor(X, lower=True, check_finite=False)
                B = cho_solve(cfac, Q, check_finite=False)  # B = X^{-1} Q
            else:  # fallback
                L = np.linalg.cholesky(X)
                B = np.linalg.solve(L.T, np.linalg.solve(L, Q))
        except np.linalg.LinAlgError:
            # Regularize slightly if ill-conditioned
            reg = 1e-12 * (np.trace(X) / d1 if d1 > 0 else 1.0)
            X_reg = X + reg * I
            if _HAVE_SCIPY:
                cfac = cho_factor(X_reg, lower=True, check_finite=False)
                B = cho_solve(cfac, Q, check_finite=False)
            else:
                L = np.linalg.cholesky(X_reg)
                B = np.linalg.solve(L.T, np.linalg.solve(L, Q))

        # Initial M_i = q_i^T X^{-1} q_i = diag(Q^T B)
        M = np.einsum("ij,ij->j", Q, B)  # length n

        for _ in range(max_iter):
            j = int(np.argmax(M))
            max_M = float(M[j])

            # Convergence: max_i M_i <= d+1 + tol
            if max_M <= d1 + tol:
                break

            # Update parameters
            gamma = (max_M - d1) / (d1 * (max_M - 1.0))
            one_minus_gamma = 1.0 - gamma
            alpha = gamma / one_minus_gamma  # α = γ/(1-γ)

            # z = X^{-1} q_j is column j of B
            z = B[:, j]  # (d1,)
            # r = Q^T z
            r = QT @ z  # (n,)
            # denom = 1 + α * δ, where δ = q_j^T z = max_M
            denom = 1.0 + alpha * max_M

            # Update B using: B_new = (1/(1-γ)) * ( B - (α/(1+α*δ)) * z r^T )
            scale_outer = alpha / denom
            s = 1.0 / one_minus_gamma
            B -= scale_outer * (z[:, None] @ r[None, :])
            B *= s

            # Update u
            u *= one_minus_gamma
            u[j] += gamma

            # Update M efficiently: M_new = s * ( M - scale_outer * r^2 )
            M = s * (M - scale_outer * (r * r))

        # Recover center and covariance-like S
        c = P @ u  # (d,)
        S = (P * u) @ P.T - np.outer(c, c)  # covariance-like
        S = 0.5 * (S + S.T)
        return c, S

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solves the minimum volume covering ellipsoid problem:
            minimize   -log det X
            s.t.       ||X a_i + Y||_2 <= 1  for all i
                       X symmetric positive semidefinite

        We compute the standard MVEE in form (x - c)^T A (x - c) <= 1 (with A ≻ 0),
        then set X = A^{1/2} (symmetric square root) and Y = -X c.

        Args:
            problem: dict with key "points": list of n points in R^d.

        Returns:
            dict with keys:
                - "objective_value": float
                - "ellipsoid": {"X": ndarray (d,d), "Y": ndarray (d,)}
        """
        pts = np.asarray(problem.get("points", []), dtype=float)
        if pts.size == 0:
            # No points: return unit ball at origin in appropriate dimension
            d = 0 if pts.ndim < 2 else pts.shape[1]
            X = np.eye(d)
            Y = np.zeros(d)
            return {"objective_value": 0.0, "ellipsoid": {"X": X, "Y": Y}}

        if pts.ndim != 2:
            pts = np.atleast_2d(pts)

        # Deduplicate identical points to reduce problem size
        try:
            pts = np.unique(pts, axis=0)
        except Exception:
            pass

        n, d = pts.shape
        P = pts.T  # shape (d, n)

        # Compute MVEE parameters (c, S)
        tol = float(kwargs.get("tol", 1e-4))
        max_iter = int(kwargs.get("max_iter", 10_000))
        c, S = self._mvee(P, tol=tol, max_iter=max_iter)

        # Eigen-decomposition of S to build X directly:
        # X = (1/sqrt(d)) * U diag(λ^{-1/2}) U^T, where S = U diag(λ) U^T
        try:
            lam, U = np.linalg.eigh(S)
        except np.linalg.LinAlgError:
            # Regularize S slightly if eigh fails
            regS = 1e-12 * (np.trace(S) / d if d > 0 else 1.0)
            lam, U = np.linalg.eigh(S + regS * np.eye(d))

        # Ensure positive eigenvalues (clip tiny negatives due to numerical issues)
        lam = np.maximum(lam, 1e-15)
        inv_sqrt_lam = 1.0 / np.sqrt(lam)
        scale = 1.0 / np.sqrt(d) if d > 0 else 1.0
        X = U @ ((scale * inv_sqrt_lam)[:, None] * U.T)
        X = 0.5 * (X + X.T)

        # Center shift
        Y = -X @ c

        # Ensure feasibility (scale down slightly if needed)
        norms = np.linalg.norm(X @ P + Y[:, None], axis=0)
        max_norm = float(np.max(norms)) if norms.size else 0.0
        if max_norm > 1.0 + 1e-10:
            X = X / max_norm
            Y = Y / max_norm

        # Objective value: -log det X computed from X (matches validator)
        sign, logdetX = np.linalg.slogdet(X)
        if not np.isfinite(logdetX) or sign <= 0:
            # Fallback via S: -log det X = 0.5*(sum log λ + d*log d)
            objective_value = float(0.5 * (np.sum(np.log(lam)) + d * np.log(d if d > 0 else 1.0)))
        else:
            objective_value = float(-logdetX)

        return {"objective_value": objective_value, "ellipsoid": {"X": X, "Y": Y}}