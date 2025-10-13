from __future__ import annotations

from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Accept both "P" and "Q" as the Hessian key
        P = np.asarray(problem.get("P", problem.get("Q")), dtype=float)
        q = np.asarray(problem["q"], dtype=float).reshape(-1)

        n = P.shape[0]
        P = 0.5 * (P + P.T)  # symmetrize

        # Helpers to robustly parse optional matrices/vectors
        def _ensure_matrix(M_key: str) -> np.ndarray:
            M = problem.get(M_key, None)
            if M is None:
                return np.zeros((0, n), dtype=float)
            M = np.asarray(M, dtype=float)
            if M.size == 0:
                return np.zeros((0, n), dtype=float)
            if M.ndim == 1:
                # Interpret 1D as a single row (one constraint)
                M = M.reshape(1, -1)
            return M

        def _ensure_vector(v_key: str, m: int) -> np.ndarray:
            v = problem.get(v_key, None)
            if v is None:
                return np.zeros((m,), dtype=float)
            v = np.asarray(v, dtype=float).reshape(-1)
            if v.size == 0:
                return np.zeros((m,), dtype=float)
            return v

        G = _ensure_matrix("G")
        h = _ensure_vector("h", G.shape[0])
        A = _ensure_matrix("A")
        b = _ensure_vector("b", A.shape[0])

        # Fast paths: unconstrained or equality-only
        m_ineq = G.shape[0]
        m_eq = A.shape[0]

        if m_ineq == 0 and m_eq == 0:
            x = self._solve_unconstrained(P, q)
            return {"solution": x.tolist()}

        if m_ineq == 0 and m_eq > 0:
            x = self._solve_eq_qp(P, q, A, b)
            return {"solution": x.tolist()}

        # General case: try OSQP directly for speed
        try:
            import osqp  # type: ignore
            from scipy import sparse  # local import to avoid overhead unless needed

            # Build combined constraint matrix/l,u:
            # Gx <= h  -> lower=-inf, upper=h
            # Ax == b  -> lower=b, upper=b
            if m_eq == 0:
                A_comb = sparse.csc_matrix(G)
                l = -np.inf * np.ones(m_ineq)
                u = h
            elif m_ineq == 0:
                A_comb = sparse.csc_matrix(A)
                l = b
                u = b
            else:
                A_comb = sparse.vstack([sparse.csc_matrix(G), sparse.csc_matrix(A)], format="csc")
                l = np.concatenate([ -np.inf * np.ones(m_ineq), b ])
                u = np.concatenate([ h, b ])

            P_csc = sparse.csc_matrix(P)

            prob = osqp.OSQP()
            prob.setup(P=P_csc, q=q, A=A_comb, l=l, u=u,
                       verbose=False, eps_abs=1e-8, eps_rel=1e-8, polish=True, max_iter=200000)
            res = prob.solve()
            # Status values: 1 solved, 2 solved inaccurate
            status_val = getattr(res.info, "status_val", None)
            if status_val in (1, 2) and res.x is not None:
                return {"solution": res.x.tolist()}
            # If polish produced a result, prefer it
            if res.x is not None:
                return {"solution": res.x.tolist()}
            # Fall through to fallback if OSQP didn't return a solution
            raise RuntimeError(f"OSQP failed with status: {getattr(res.info, 'status', 'unknown')}")
        except Exception:
            # Fallback: use CVXPY with OSQP; close to reference but attempted only if OSQP not available
            try:
                import cvxpy as cp  # type: ignore

                x_var = cp.Variable(n)
                objective = 0.5 * cp.quad_form(x_var, cp.psd_wrap(P)) + q @ x_var
                constraints = []
                if m_ineq > 0:
                    constraints.append(G @ x_var <= h)
                if m_eq > 0:
                    constraints.append(A @ x_var == b)
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, warm_start=True, verbose=False)
                if x_var.value is None:
                    raise ValueError("CVXPY failed to produce a solution.")
                return {"solution": x_var.value.tolist()}
            except Exception:
                # Last-resort numerical fallback using SciPy trust-constr
                from scipy.optimize import minimize, LinearConstraint  # type: ignore

                def fun(x: np.ndarray) -> float:
                    return 0.5 * x.dot(P).dot(x) + q.dot(x)

                def jac(x: np.ndarray) -> np.ndarray:
                    return P.dot(x) + q

                constraints = []
                if m_ineq > 0:
                    constraints.append(LinearConstraint(G, -np.inf * np.ones(m_ineq), h))
                if m_eq > 0:
                    constraints.append(LinearConstraint(A, b, b))

                x0 = np.zeros(n, dtype=float)
                res = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    jac=jac,
                    hess=lambda x: P,
                    constraints=constraints,
                    options={"xtol": 1e-12, "gtol": 1e-12, "barrier_tol": 1e-12, "verbose": 0, "maxiter": 1000},
                )
                x = res.x if res.success and res.x is not None else x0
                return {"solution": x.tolist()}

    @staticmethod
    def _solve_unconstrained(P: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Solve min 0.5 x^T P x + q^T x -> P x = -q
        n = P.shape[0]
        try:
            # Try a direct solve first (works if P is PD)
            return np.linalg.solve(P, -q)
        except Exception:
            # Use pseudoinverse for PSD/singular cases (minimal-norm solution)
            # If problem is unbounded, this returns one minimizer if it exists
            return -np.linalg.pinv(P, rcond=1e-12) @ q

    @staticmethod
    def _solve_eq_qp(P: np.ndarray, q: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Solve: min 0.5 x^T P x + q^T x s.t. A x = b
        n = P.shape[0]
        # Find feasible x0 (least squares). For consistent systems, residual ~ 0.
        x0, *_ = np.linalg.lstsq(A, b, rcond=None)
        # Nullspace of A via SVD
        # A = U S V^T; Null(A) spanned by columns of V corresponding to zero singular values
        U, S, Vt = np.linalg.svd(A, full_matrices=True)
        tol = np.finfo(float).eps * max(A.shape) * (S[0] if S.size > 0 else 1.0)
        r = (S > tol).sum()
        if r >= n:
            # No degrees of freedom; x0 is (nearly) unique feasible
            return x0[:n] if x0.shape[0] != n else x0
        V = Vt.T
        Z = V[:, r:]  # n x (n-r)
        # Reduce to y variables: x = x0 + Z y
        Px0_q = P @ x0 + q
        H = Z.T @ (P @ Z)  # PSD
        g = Z.T @ Px0_q
        # Solve H y = -g (use pinv for robustness if singular)
        try:
            y = np.linalg.solve(H, -g)
        except Exception:
            y = -np.linalg.pinv(H, rcond=1e-12) @ g
        x = x0 + Z @ y
        # Ensure the length is n
        return x[:n] if x.shape[0] != n else x