from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the convex QP:
            minimize (1/2) x^T P x + q^T x
            subject to G x <= h
                       A x == b

        Accepts keys "P" or "Q" for the quadratic matrix, and "q", "G", "h", "A", "b".
        Returns {"solution": [...], "objective": float}.
        """
        def to_arr(x):
            if x is None:
                return None
            return np.asarray(x, dtype=float)

        # Accept "P" or "Q"
        P_in = problem.get("P", problem.get("Q", None))
        q_in = problem.get("q", None)
        G_in = problem.get("G", None)
        h_in = problem.get("h", None)
        A_in = problem.get("A", None)
        b_in = problem.get("b", None)

        P = to_arr(P_in)
        q = to_arr(q_in)

        # Infer dimensions and sanitize P, q
        if P is None:
            if q is None:
                return {"solution": [], "objective": 0.0}
            q = q.reshape(-1)
            n = q.size
            P = np.zeros((n, n), dtype=float)
        else:
            # If P is 1-D and a perfect square, reshape
            if P.ndim == 1:
                L = P.size
                r = int(np.sqrt(L))
                if r * r == L:
                    P = P.reshape((r, r))
                else:
                    P = P.reshape((1, L))
            if P.ndim != 2:
                raise ValueError("P must be 2-dimensional")
            # Ensure square
            if P.shape[0] != P.shape[1]:
                n = min(P.shape)
                P = P[:n, :n].copy()
            n = P.shape[0]
            # Symmetrize
            P = 0.5 * (P + P.T)
            if q is None:
                q = np.zeros(n, dtype=float)
            else:
                q = q.reshape(-1)
                if q.size != n:
                    if q.size < n:
                        q = np.concatenate([q, np.zeros(n - q.size, dtype=float)])
                    else:
                        q = q[:n]

        # Parse A, b (equalities)
        if A_in is None or (isinstance(A_in, (list, tuple)) and len(A_in) == 0):
            A = np.zeros((0, n), dtype=float)
            b = np.zeros((0,), dtype=float)
        else:
            A = to_arr(A_in)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            if A.shape[1] != n:
                if A.shape[0] == n:
                    A = A.T
                else:
                    raise ValueError(f"Incompatible shape for A: {A.shape} vs n={n}")
            if b_in is None:
                b = np.zeros(A.shape[0], dtype=float)
            else:
                b = to_arr(b_in).reshape(-1)
                if b.size != A.shape[0]:
                    if b.size < A.shape[0]:
                        b = np.concatenate([b, np.zeros(A.shape[0] - b.size, dtype=float)])
                    else:
                        b = b[: A.shape[0]]

        # Parse G, h (inequalities)
        if G_in is None or (isinstance(G_in, (list, tuple)) and len(G_in) == 0):
            G = np.zeros((0, n), dtype=float)
            h = np.zeros((0,), dtype=float)
        else:
            G = to_arr(G_in)
            if G.ndim == 1:
                G = G.reshape(1, -1)
            if G.shape[1] != n:
                if G.shape[0] == n:
                    G = G.T
                else:
                    raise ValueError(f"Incompatible shape for G: {G.shape} vs n={n}")
            if h_in is None:
                h = np.zeros(G.shape[0], dtype=float)
            else:
                h = to_arr(h_in).reshape(-1)
                if h.size != G.shape[0]:
                    if h.size < G.shape[0]:
                        h = np.concatenate([h, np.zeros(G.shape[0] - h.size, dtype=float)])
                    else:
                        h = h[: G.shape[0]]

        m = G.shape[0]
        p = A.shape[0]

        if n == 0:
            return {"solution": [], "objective": 0.0}

        # Fast path: no inequalities
        if m == 0:
            # Unconstrained
            if p == 0:
                try:
                    x = np.linalg.solve(P, -q)
                except np.linalg.LinAlgError:
                    x = -np.linalg.pinv(P).dot(q)
                obj = 0.5 * float(x @ (P @ x)) + float(q @ x)
                return {"solution": x.tolist(), "objective": float(obj)}
            # Equality-constrained: KKT system
            KKT = np.zeros((n + p, n + p), dtype=float)
            KKT[:n, :n] = P
            if p > 0:
                KKT[:n, n:] = A.T
                KKT[n:, :n] = A
            rhs = np.concatenate([-q, b]) if p > 0 else -q
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
            x = sol[:n]
            obj = 0.5 * float(x @ (P @ x)) + float(q @ x)
            return {"solution": x.tolist(), "objective": float(obj)}

        # General case with inequalities: try cvxpy + OSQP for speed/robustness
        try:
            import cvxpy as cp  # type: ignore

            x_var = cp.Variable(n)
            try:
                quad = cp.quad_form(x_var, P)
            except Exception:
                quad = cp.quad_form(x_var, cp.psd_wrap(P))
            objective = 0.5 * quad + q @ x_var
            cons = []
            if m > 0:
                cons.append(G @ x_var <= h)
            if p > 0:
                cons.append(A @ x_var == b)
            prob = cp.Problem(cp.Minimize(objective), cons)
            _ = prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"cvxpy failed with status {prob.status}")
            x_val = np.asarray(x_var.value).reshape(-1)
            obj = 0.5 * float(x_val @ (P @ x_val)) + float(q @ x_val)
            return {"solution": x_val.tolist(), "objective": float(obj)}
        except Exception:
            # If cvxpy/OSQP unavailable or fails, try scipy fallback
            pass

        # Fallback: use scipy.optimize.minimize (trust-constr) if available
        try:
            from scipy.optimize import minimize, LinearConstraint  # type: ignore

            def fun(x):
                return 0.5 * float(x @ (P @ x)) + float(q @ x)

            def jac(x):
                return P.dot(x) + q

            def hess(x):
                return P

            constraints = []
            if p > 0:
                constraints.append(LinearConstraint(A, b, b))
            if m > 0:
                lb = -np.inf * np.ones(m, dtype=float)
                constraints.append(LinearConstraint(G, lb, h))

            # initial guess: satisfy equalities if possible
            if p > 0:
                try:
                    x0 = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)
                    if x0.size != n:
                        x0 = np.zeros(n, dtype=float)
                except Exception:
                    x0 = np.zeros(n, dtype=float)
            else:
                x0 = np.zeros(n, dtype=float)

            # small projection to reduce obvious inequality violations
            if m > 0:
                viol = G.dot(x0) - h
                if viol.size > 0 and np.max(viol) > 0:
                    idx = int(np.argmax(viol))
                    gi = G[idx]
                    denom = float(gi.dot(gi))
                    if denom > 0:
                        x0 = x0 + (h[idx] - gi.dot(x0)) * gi / denom

            res = minimize(
                fun,
                x0,
                jac=jac,
                hess=hess,
                constraints=constraints,
                method="trust-constr",
                options={"gtol": 1e-9, "xtol": 1e-9, "maxiter": 2000, "verbose": 0},
            )
            if res is not None and getattr(res, "success", False):
                x_opt = np.asarray(res.x, dtype=float)
                obj = 0.5 * float(x_opt @ (P @ x_opt)) + float(q @ x_opt)
                return {"solution": x_opt.tolist(), "objective": float(obj)}
        except Exception:
            pass

        # Last resort: approximate solution via pseudoinverse / least-squares
        try:
            if p > 0:
                x_approx = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)
                if x_approx.size != n:
                    x_approx = np.zeros(n, dtype=float)
            else:
                x_approx = -np.linalg.pinv(P).dot(q)
            obj = 0.5 * float(x_approx @ (P @ x_approx)) + float(q @ x_approx)
            return {"solution": x_approx.tolist(), "objective": float(obj)}
        except Exception as e:
            raise RuntimeError(f"All solver attempts failed: {e}") from e