import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, list]:
        """
        Solves the LP centering problem using a dual Newton method.
        The problem is assumed to be the standard LP centering problem:
            minimize c^T*x - sum(log(x_i))
            subject to Ax = b, x > 0.

        The method solves the corresponding dual problem, which is unconstrained
        and convex, and then recovers the primal solution.
        """
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)

        if A.ndim == 1:
            A = A.reshape(1, -1)

        m, n = A.shape

        # --- Phase 1: Find a strictly feasible dual starting point v ---
        # The dual problem is: minimize h(v) = b^T*v - sum(log(c + A^T*v))
        # We need a starting v such that the log argument is positive: c + A^T*v > 0.
        # We can find such a v by solving a simple LP:
        #   maximize t
        #   subject to c + A^T*v >= t  =>  -A^T*v + t*1 <= c
        c_lp = np.zeros(m + 1)
        c_lp[-1] = -1.0
        A_lp = np.hstack((-A.T, np.ones((n, 1))))
        b_lp = c

        res = linprog(c=c_lp, A_ub=A_lp, b_ub=b_lp, bounds=(None, None), method='highs')

        if not res.success or -res.fun < 1e-8:
            # Fallback heuristic if LP fails: solve A^T*v = -c + 1
            try:
                v = np.linalg.lstsq(A.T, -c + 1, rcond=None)[0]
                if np.any(c + A.T @ v <= 1e-8):
                    v = np.linalg.pinv(A.T) @ (-c + 1)
            except np.linalg.LinAlgError:
                v = np.zeros(m)
        else:
            v = res.x[:-1]

        # --- Phase 2: Newton's method for the dual problem ---
        # Minimize h(v) = b^T*v - sum(log(c + A^T*v))
        max_iter = 50
        tol = 1e-9
        alpha = 0.25  # Armijo condition parameter
        beta = 0.5    # Backtracking line search reduction factor
        x_sol = None

        for _ in range(max_iter):
            s = c + A.T @ v

            if np.any(s <= 1e-12):
                break

            x_sol = 1.0 / s
            
            grad = b - A @ x_sol

            H = (A * x_sol**2) @ A.T + 1e-12 * np.eye(m)

            try:
                L = np.linalg.cholesky(H)
                w = np.linalg.solve(L, -grad)
                dv = np.linalg.solve(L.T, w)
            except np.linalg.LinAlgError:
                dv = np.linalg.lstsq(H, -grad, rcond=None)[0]

            lambda_sq = -grad.T @ dv
            if lambda_sq / 2.0 < tol:
                break

            t = 1.0
            h_v = b.T @ v - np.sum(np.log(s))
            grad_dot_dv = grad.T @ dv

            while True:
                v_new = v + t * dv
                s_new = c + A.T @ v_new
                
                if np.all(s_new > 0):
                    h_v_new = b.T @ v_new - np.sum(np.log(s_new))
                    if h_v_new <= h_v + alpha * t * grad_dot_dv:
                        break
                
                t *= beta
                if t < 1e-14:
                    t = 0.0
                    break

            if t == 0.0:
                break
            
            v = v + t * dv

        if x_sol is None:
            s = c + A.T @ v
            s[s <= 1e-12] = 1e-12
            x_sol = 1.0 / s

        return {"solution": x_sol.tolist()}