import numpy as np
import cvxpy as cp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the long‑only Markowitz portfolio optimization problem.

        maximize   μᵀ w - γ * wᵀ Σ w
        subject to 1ᵀ w = 1,  w ≥ 0
        """
        # Prepare problem data
        mu = np.asarray(problem["μ"], dtype=float)
        Sigma = np.asarray(problem["Σ"], dtype=float)
        gamma = float(problem["γ"])
        n = mu.size

        # Solve the long‑only Markowitz portfolio optimization problem using OSQP directly.
        # Formulate as a quadratic program:
        #   minimize   (γ) * wᵀ Σ w  -  μᵀ w
        #   subject to 1ᵀ w = 1,  w ≥ 0
        #
        # Standard QP form for OSQP:
        #   minimize   (1/2) wᵀ P w + qᵀ w
        #   subject to  l ≤ A w ≤ u
        #
        # where P = 2γ Σ, q = -μ,
        # A = [I; 1ᵀ],   l = [0,…,0, 1],   u = [∞,…,∞, 1].
        try:
            import osqp
            import scipy.sparse as sp

            # Build OSQP data
            P = 2.0 * gamma * Sigma
            # Ensure symmetry
            P = (P + P.T) / 2.0
            P_csc = sp.csc_matrix(P)

            q = -mu

            # Constraint matrix: first n rows enforce w ≥ 0, last row enforces sum(w) = 1
            A = sp.vstack([sp.eye(n), sp.csc_matrix(np.ones((1, n)))])
            l = np.hstack([np.zeros(n), 1.0])
            u = np.hstack([np.full(n, np.inf), 1.0])

            # OSQP solver
            prob = osqp.OSQP()
            prob.setup(P=P_csc, q=q, A=A, l=l, u=u,
                       verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                       warm_start=True)
            res = prob.solve()
            w_opt = res.x
        except Exception:
            # Fallback to CVXPY with ECOS solver if OSQP is unavailable
            w = cp.Variable(n)
            objective = cp.Minimize(-mu @ w + gamma * cp.quad_form(w, cp.psd_wrap(Sigma)))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, warm_start=True)
            w_opt = w.value

        # Post‑process solution: clip tiny negatives and renormalise to sum to 1
        if w_opt is None or not np.isfinite(w_opt).all():
            w_opt = np.full(n, 1.0 / n)
        else:
            w_opt = np.clip(w_opt, 0.0, None)
            total = w_opt.sum()
            if total > 0:
                w_opt /= total
            else:
                w_opt = np.full(n, 1.0 / n)

        return {"w": w_opt.tolist()}