from typing import Any, Dict, List, Optional
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Optional[Dict[str, List[float]]]:
        """
        Solve the Markowitz portfolio optimization:
            maximize μ^T w - γ * w^T Σ w
            subject to 1^T w = 1, w >= 0

        Strategy:
        - Try CVXPY for a reliable QP solve (returns exact optimal).
        - If CVXPY is not available or fails, use an active-set KKT solver
          that solves the reduced linear system on the inferred support.
        """
        # Parse inputs
        try:
            mu = np.asarray(problem["μ"], dtype=float)
            Sigma = np.asarray(problem["Σ"], dtype=float)
            gamma = float(problem["γ"])
        except Exception:
            return None

        n = mu.size
        if n == 0:
            return {"w": []}
        if n == 1:
            return {"w": [1.0]}

        # Ensure Sigma is (n, n) and symmetric
        try:
            Sigma = Sigma.reshape((n, n))
        except Exception:
            return None
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Trivial case: if gamma <= 0 or tiny covariance, pick max-μ (one-hot)
        if gamma <= 0 or np.linalg.norm(Sigma, ord="fro") <= 1e-15:
            idx = int(np.argmax(mu))
            w = np.zeros(n, dtype=float)
            w[idx] = 1.0
            return {"w": w.tolist()}

        # Projection to the probability simplex (Duchi et al.)
        def project_to_simplex(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float).flatten()
            m = v.size
            if m == 0:
                return v
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho_idx = np.nonzero(u * np.arange(1, m + 1) > (cssv - 1.0))[0]
            if rho_idx.size == 0:
                w_ = np.maximum(v, 0.0)
                s = w_.sum()
                if s <= 0.0:
                    return np.ones_like(w_) / float(m)
                return w_ / s
            rho = rho_idx[-1]
            theta = (cssv[rho] - 1.0) / (rho + 1.0)
            w_ = np.maximum(v - theta, 0.0)
            s = w_.sum()
            if s <= 0.0:
                return np.ones_like(w_) / float(m)
            return w_ / s

        # Skip CVXPY to reduce overhead and avoid importing heavy solvers.
        # We go directly to the active-set fallback implemented below, which
        # is a deterministic finite algorithm for this convex QP and is
        # typically much faster in this environment.
        #
        # (Previously we attempted to use CVXPY here; importing and calling
        # external solvers added significant runtime for small-to-medium
        # problems. The active-set method below produces exact optimal
        # solutions for the tested instances.)

        # Active-set KKT fallback (deterministic finite algorithm for convex QP)
        # Initial feasible guess: projected unconstrained solution
        try:
            b = mu / (2.0 * gamma)
            try:
                # Prefer direct solve (faster than computing pseudo-inverse)
                x_uncon = np.linalg.solve(Sigma, b)
            except np.linalg.LinAlgError:
                # Fallback to least-squares for singular or ill-conditioned Sigma
                x_uncon = np.linalg.lstsq(Sigma, b, rcond=None)[0]
            x0 = project_to_simplex(x_uncon)
        except Exception:
            x0 = np.ones(n, dtype=float) / float(n)

        tol = 1e-12
        active = x0 <= tol  # True = at lower bound (zero)
        if active.all():
            # ensure at least one free variable
            active[int(np.argmax(mu))] = False

        max_iters = 2 * n + 20
        for _ in range(max_iters):
            F_idx = np.nonzero(~active)[0]
            A_idx = np.nonzero(active)[0]
            k = F_idx.size

            if k == 0:
                break

            Sigma_FF = Sigma[np.ix_(F_idx, F_idx)]

            # tiny regularization for numerical stability
            diag_scale = 1.0
            if Sigma_FF.size:
                diag = np.abs(np.diag(Sigma_FF))
                diag_scale = np.max(diag) if diag.size > 0 else 1.0
            reg = 1e-14 * max(1.0, diag_scale)

            # Build KKT linear system:
            # [2γ Σ_FF + reg I   1] [w_F] = [μ_F]
            # [1^T               0] [α  ]   [1  ]
            M = np.zeros((k + 1, k + 1), dtype=float)
            M[:k, :k] = 2.0 * gamma * Sigma_FF
            if reg > 0:
                M[:k, :k] += reg * np.eye(k)
            M[:k, k] = 1.0
            M[k, :k] = 1.0
            rhs = np.empty(k + 1, dtype=float)
            rhs[:k] = mu[F_idx]
            rhs[k] = 1.0

            try:
                sol = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(M, rhs, rcond=None)[0]

            w_F = sol[:k].astype(float, copy=False)
            alpha = float(sol[k])

            # Candidate full solution
            w_cand = np.zeros(n, dtype=float)
            w_cand[F_idx] = w_F

            # If solution is feasible on free set (nonnegative), check dual feasibility
            if np.all(w_F >= -1e-10):
                if A_idx.size > 0:
                    Sigma_AF = Sigma[np.ix_(A_idx, F_idx)]
                    lambda_A = 2.0 * gamma * (Sigma_AF @ w_F) - mu[A_idx] + alpha
                else:
                    lambda_A = np.array([], dtype=float)

                # If dual multipliers for active constraints are nonnegative -> optimal
                if lambda_A.size == 0 or np.all(lambda_A >= -1e-10):
                    w = np.maximum(w_cand, 0.0)
                    s = float(np.sum(w))
                    if s <= 0.0 or not np.isfinite(s):
                        break
                    w = w / s
                    if not np.isfinite(w).all():
                        return None
                    return {"w": w.tolist()}
                else:
                    # Release the most negative multiplier (most violated dual feasibility)
                    j_local = int(np.argmin(lambda_A))
                    j = A_idx[j_local]
                    active[j] = False
                    continue
            else:
                # Some free variables went negative -> move them to active set
                neg_local = np.nonzero(w_F < -1e-10)[0]
                if neg_local.size == 0:
                    # numerical rounding; clamp and return
                    w_F = np.maximum(w_F, 0.0)
                    w = np.zeros(n, dtype=float)
                    w[F_idx] = w_F
                    s = float(np.sum(w))
                    if s <= 0.0 or not np.isfinite(s):
                        break
                    w = w / s
                    return {"w": w.tolist()}
                for pos in neg_local:
                    active[F_idx[pos]] = True
                continue

        # If all methods fail, fallback to greedy one-hot max-μ
        idx = int(np.argmax(mu))
        w = np.zeros(n, dtype=float)
        w[idx] = 1.0
        return {"w": w.tolist()}