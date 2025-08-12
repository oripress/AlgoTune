import numpy as np
from typing import Any, Dict

# Optional CVXPY import for correctness (preferred)
try:
    import cvxpy as cp
except Exception:
    cp = None

# Avoid invoking CVXPY inside solve to save heavy overhead; prefer the fast Newton fallback.
# (Do not forcibly disable the cvxpy import above — allow using CVXPY when it's available so the
# solver matches the reference implementation's results.)
class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Dict[str, list]:
        """
        Solve the LP-centering problem:
            minimize    c^T x - sum_i log(x_i)
            subject to  Ax = b,  x > 0

        Strategy:
         - Prefer CVXPY (use CLARABEL if installed, otherwise best available).
         - If CVXPY produces a solution, refine it via a high-accuracy Newton method on the dual
           to ensure the primal feasibility/optimality matches the reference within tight tolerances.
         - If CVXPY is unavailable or fails, use a robust damped-Newton method on the dual variable y
           solving g(y) = A (1/(c + A^T y)) - b = 0.
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # Normalize A shape
        if A.size == 0:
            m = 0
            n = c.size
            A = A.reshape((0, n))
        else:
            if A.ndim == 1:
                A = A.reshape((1, -1))
            m, n = A.shape

        # Trivial cases
        if n == 0:
            return {"solution": []}

        if m == 0:
            # Unconstrained separable: derivative c - 1/x = 0 -> x = 1/c for c>0
            x = np.empty_like(c)
            pos = c > 0
            x[pos] = 1.0 / c[pos]
            x[~pos] = 1e6
            return {"solution": x.tolist()}

        At = A.T

        def is_valid_primal(x_vec):
            if x_vec is None:
                return False
            x_arr = np.asarray(x_vec, dtype=float)
            if not np.isfinite(x_arr).all():
                return False
            if np.any(x_arr <= 0):
                return False
            if np.linalg.norm(A.dot(x_arr) - b) > 1e-6 * max(1.0, np.linalg.norm(b)):
                return False
            return True

        def find_initial_y(cvec, A_mat):
            """
            Try multiple deterministic strategies to find y with s = c + A^T y > 0.
            """
            At_loc = A_mat.T
            m_loc = A_mat.shape[0]
            eps_list = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
            for eps in eps_list:
                g = np.maximum(-cvec + eps, 0.0)
                try:
                    y, *_ = np.linalg.lstsq(At_loc, g, rcond=None)
                except Exception:
                    y = np.zeros(m_loc, dtype=float)
                s = cvec + At_loc.dot(y)
                if np.all(s > 0):
                    return y
            # Deterministic RNG attempts
            rng = np.random.default_rng(0)
            for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                for _ in range(50):
                    y_try = rng.normal(scale=scale, size=m_loc)
                    s = cvec + At_loc.dot(y_try)
                    if np.all(s > 0):
                        return y_try
            # Fallback zero
            return np.zeros(m_loc, dtype=float)

        def newton_refine_from_x(x_init, cvec, A_mat, bvec, max_iters=80):
            """
            Given a primal iterate x_init (positive), compute y0 from least squares
            A^T y0 ~= 1/x_init - c and refine y via Newton to solve g(y) = A(1/(c+A^T y)) - b = 0.
            Return refined x (or None on failure).
            """
            x_init = np.asarray(x_init, dtype=float)
            if x_init is None or np.any(x_init <= 0) or not np.isfinite(x_init).all():
                return None
            s0 = 1.0 / x_init
            # Solve A^T y = s0 - c in least squares sense
            try:
                y0, *_ = np.linalg.lstsq(A_mat.T, s0 - cvec, rcond=None)
            except Exception:
                y0 = find_initial_y(cvec, A_mat)

            y = y0.astype(float)
            bnorm = max(1.0, np.linalg.norm(bvec))
            for _ in range(max_iters):
                s = cvec + A_mat.T.dot(y)
                if np.any(s <= 0) or not np.isfinite(s).all():
                    # Try to recover
                    y = find_initial_y(cvec, A_mat)
                    s = cvec + A_mat.T.dot(y)
                    if np.any(s <= 0):
                        return None
                x = 1.0 / s
                g = A_mat.dot(x) - bvec
                gnorm = np.linalg.norm(g)
                if gnorm <= 1e-12 * bnorm:
                    return x
                D = x * x  # x^2
                AD = A_mat * D  # shape m x n broadcasting columns
                M = AD.dot(A_mat.T)
                # robust regularization
                diag_max = np.max(np.abs(np.diag(M))) if M.size else 0.0
                reg = max(1e-16 * max(diag_max, 1.0), 1e-14)
                M_reg = M + reg * np.eye(M.shape[0])
                try:
                    delta = np.linalg.solve(M_reg, g)
                except Exception:
                    delta, *_ = np.linalg.lstsq(M_reg, g, rcond=None)
                # Backtracking
                alpha = 1.0
                accepted = False
                phi0 = 0.5 * (gnorm ** 2)
                for _bt in range(40):
                    y_new = y + alpha * delta
                    s_new = cvec + A_mat.T.dot(y_new)
                    if np.any(s_new <= 0) or not np.isfinite(s_new).all():
                        alpha *= 0.5
                        continue
                    x_new = 1.0 / s_new
                    g_new = A_mat.dot(x_new) - bvec
                    gnorm_new = np.linalg.norm(g_new)
                    phi_new = 0.5 * (gnorm_new ** 2)
                    if phi_new <= phi0 - 1e-8 * alpha * (gnorm ** 2) or gnorm_new < gnorm * 0.999999:
                        y = y_new
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    y = y + alpha * delta
            # Final check
            s = cvec + A_mat.T.dot(y)
            if np.any(s <= 0) or not np.isfinite(s).all():
                return None
            x_final = 1.0 / s
            if is_valid_primal(x_final):
                return x_final
            return None

        # Try CVXPY first (if available). Prefer CLARABEL, otherwise pick best available solver.
        if cp is not None:
            try:
                x_var = cp.Variable(n)
                objective = cp.Minimize(c @ x_var - cp.sum(cp.log(x_var)))
                constraints = [A @ x_var == b]
                prob = cp.Problem(objective, constraints)

                # Choose solver intelligently
                try:
                    installed = set(cp.installed_solvers())
                except Exception:
                    installed = set()

                chosen = None
                for cand in ("CLARABEL", "ECOS", "SCS", "OSQP", "CVXOPT"):
                    if cand in installed:
                        chosen = cand
                        break

                # Try chosen solver with safe options
                solved = False
                if chosen == "CLARABEL":
                    try:
                        prob.solve(solver="CLARABEL", verbose=False)
                        solved = True
                    except Exception:
                        solved = False
                elif chosen == "ECOS":
                    try:
                        prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, verbose=False, max_iters=5000)
                        solved = True
                    except Exception:
                        try:
                            prob.solve(solver=cp.ECOS, verbose=False)
                            solved = True
                        except Exception:
                            solved = False
                elif chosen == "SCS":
                    try:
                        prob.solve(solver=cp.SCS, eps=1e-6, max_iters=200000, verbose=False)
                        solved = True
                    except Exception:
                        try:
                            prob.solve(solver=cp.SCS, verbose=False)
                            solved = True
                        except Exception:
                            solved = False
                elif chosen == "OSQP":
                    try:
                        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
                        solved = True
                    except Exception:
                        try:
                            prob.solve(solver=cp.OSQP, verbose=False)
                            solved = True
                        except Exception:
                            solved = False
                else:
                    # Default attempt
                    try:
                        prob.solve(verbose=False)
                        solved = True
                    except Exception:
                        solved = False

                # Extract and refine CVXPY solution if present
                x_val = None
                try:
                    x_val = x_var.value
                except Exception:
                    x_val = None

                if x_val is not None:
                    x_val = np.asarray(x_val, dtype=float).reshape(-1)
                    # Clip tiny negatives
                    x_val = np.where(np.abs(x_val) < 1e-16, np.abs(x_val), x_val)
                    # If primal is already valid, return it
                    if is_valid_primal(x_val):
                        return {"solution": x_val.tolist()}
                    # Attempt to refine obtained x via Newton on dual (should converge quickly if x is close)
                    refined = newton_refine_from_x(x_val, c, A, b, max_iters=120)
                    if refined is not None:
                        return {"solution": refined.tolist()}
                    # As a last resort, clip to positive and return if feasible enough
                    x_clip = np.maximum(x_val, 1e-16)
                    if is_valid_primal(x_clip):
                        return {"solution": x_clip.tolist()}
                # If CVXPY produced nothing usable, fallthrough to Newton fallback
            except Exception:
                # any CVXPY-related error -> fallback to Newton
                pass

        # Fallback: robust damped Newton on dual y from scratch
        # Initialize y so that s = c + A^T y > 0
        y = np.zeros(m, dtype=float)
        s = c + At.dot(y)
        if s.min() <= 0:
            y = find_initial_y(c, A)
            s = c + At.dot(y)

        max_iters = 200
        tol = 1e-12
        b_norm = max(1.0, np.linalg.norm(b))

        for _ in range(max_iters):
            s = c + At.dot(y)
            if np.any(s <= 0) or not np.isfinite(s).all():
                y = find_initial_y(c, A)
                s = c + At.dot(y)
                if np.any(s <= 0):
                    # tiny perturbation
                    y = y + 1e-12
                    s = c + At.dot(y)

            x = 1.0 / s
            g = A.dot(x) - b
            gnorm = np.linalg.norm(g)
            if gnorm <= max(tol, 1e-12) * b_norm:
                break

            D = x * x
            AD = A * D
            M = AD.dot(A.T)

            diag_max = np.max(np.abs(np.diag(M))) if m > 0 else 0.0
            reg = max(1e-14 * max(diag_max, 1.0), 1e-14)
            M_reg = M + reg * np.eye(m)

            try:
                delta = np.linalg.solve(M_reg, g)
            except Exception:
                delta, *_ = np.linalg.lstsq(M_reg, g, rcond=None)

            # Backtracking line search
            alpha = 1.0
            phi0 = 0.5 * (gnorm ** 2)
            accepted = False
            for _bt in range(60):
                y_new = y + alpha * delta
                s_new = c + At.dot(y_new)
                if np.any(s_new <= 0) or not np.isfinite(s_new).all():
                    alpha *= 0.5
                    continue
                x_new = 1.0 / s_new
                g_new = A.dot(x_new) - b
                gnorm_new = np.linalg.norm(g_new)
                phi_new = 0.5 * (gnorm_new ** 2)
                if phi_new <= phi0 - 1e-8 * alpha * (gnorm ** 2) or gnorm_new < gnorm * 0.999999:
                    y = y_new
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                y = y + alpha * delta
                if np.any(c + At.dot(y) <= 0):
                    y = find_initial_y(c, A)

        s = c + At.dot(y)
        x = 1.0 / s
        x = np.asarray(x, dtype=float)

        # Sanitize numerical values
        x[np.isnan(x)] = 1e6
        x[np.isinf(x)] = 1e6
        x = np.maximum(x, 1e-16)

        # Final check
        if is_valid_primal(x):
            return {"solution": x.tolist()}

        # Last-resort: attempt CVXPY again if available (without raising)
        if cp is not None:
            try:
                x_var = cp.Variable(n)
                objective = cp.Minimize(c @ x_var - cp.sum(cp.log(x_var)))
                constraints = [A @ x_var == b]
                prob = cp.Problem(objective, constraints)
                try:
                    prob.solve(solver="CLARABEL", verbose=False)
                except Exception:
                    try:
                        prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, verbose=False, max_iters=5000)
                    except Exception:
                        prob.solve(verbose=False)
                x_val = x_var.value
                if x_val is not None:
                    x_val = np.asarray(x_val, dtype=float).reshape(-1)
                    x_val = np.maximum(x_val, 1e-16)
                    if is_valid_primal(x_val):
                        return {"solution": x_val.tolist()}
            except Exception:
                pass

        # Fallback least-squares positive projection
        try:
            x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
            x_ls = np.asarray(x_ls, dtype=float)
            x_ls = np.maximum(x_ls, 1e-8)
            return {"solution": x_ls.tolist()}
        except Exception:
            x[np.isnan(x)] = 1e6
            x[np.isinf(x)] = 1e6
            x = np.maximum(x, 1e-12)
            return {"solution": x.tolist()}