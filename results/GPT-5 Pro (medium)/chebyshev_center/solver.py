from typing import Any, Dict, List

import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve the Chebyshev center problem as a linear program.

        maximize    r
        subject to  a_i^T x + r * ||a_i||_2 <= b_i, for i = 1..m
                    r >= 0

        This is formulated for SciPy's linprog (HiGHS).

        :param problem: A dictionary with keys:
                        - "a": 2D array-like of shape (m, n)
                        - "b": 1D array-like of length m
        :return: {"solution": list of length n} the center x_c of the largest inscribed ball.
        """
        from scipy.optimize import linprog

        A = np.asarray(problem["a"], dtype=float)
        b = np.asarray(problem["b"], dtype=float).reshape(-1)

        # Ensure A is 2D
        if A.ndim == 1:
            A = A.reshape(1, -1)
        m, n = A.shape
        if b.size != m:
            raise ValueError(f"Shape mismatch: A has {m} rows but b has length {b.size}.")

        # Row norms ||a_i||_2
        s = np.linalg.norm(A, axis=1)

        # LP variables: v = [x (n vars), r]
        # Objective: maximize r  <=> minimize -r
        c = np.zeros(n + 1, dtype=float)
        c[-1] = -1.0

        # Inequality constraints: A @ x + s * r <= b
        A_ub = np.hstack((A, s[:, None]))
        b_ub = b

        # Bounds: x free, r >= 0
        bounds = [(None, None)] * n + [(0.0, None)]

        # Solve via HiGHS (fast and reliable)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            # Try alternative HiGHS algorithms, then gracefully fallback to CVXPY if available.
            for method in ("highs-ipm", "highs-ds"):
                try:
                    res_alt = linprog(
                        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method
                    )
                except Exception:
                    res_alt = None
                if res_alt is not None and res_alt.success:
                    res = res_alt
                    break
            else:
                # As a last resort, use CVXPY if available
                try:
                    import cvxpy as cp

                    x_var = cp.Variable(n)
                    r_var = cp.Variable(nonneg=True)
                    prob = cp.Problem(cp.Maximize(r_var), [A @ x_var + r_var * s <= b_ub])
                    prob.solve(solver="CLARABEL")
                    if prob.status in {"optimal", "optimal_inaccurate"} and x_var.value is not None:
                        x_val = np.asarray(x_var.value, dtype=float).reshape(-1)
                        if np.all(np.isfinite(x_val)):
                            return {"solution": x_val.tolist()}
                except Exception:
                    pass
                raise RuntimeError(f"Linear program failed to solve: {res.message}")

        v = res.x
        x_sol = v[:n]
        if not np.all(np.isfinite(x_sol)):
            raise RuntimeError("Non-finite solution returned by LP solver.")
        return {"solution": x_sol.tolist()}