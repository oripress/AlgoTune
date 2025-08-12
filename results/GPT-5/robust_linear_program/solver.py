from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solves a given robust LP (ellipsoidal uncertainty) as an SOCP:
            minimize    c^T x
            subject to  ||P_i^T x||_2 <= b_i - q_i^T x  for all i

        Args:
            problem: A dictionary with keys:
                - "c": list/array of length n
                - "b": list/array of length m
                - "P": list/array of shape (m, n, n)
                - "q": list/array of shape (m, n)

        Returns:
            dict with keys:
                - "objective_value": float
                - "x": np.ndarray of shape (n,)
        """
        # Parse inputs to numpy arrays
        c = np.asarray(problem["c"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)

        m = len(P)
        n = int(c.shape[0])

        # Decision variable
        x = cp.Variable(n)

        # Build SOC constraints: ||P_i^T x||_2 <= b_i - q_i^T x
        constraints = []
        for i in range(m):
            # Right-hand side scalar
            rhs = b[i] - q[i] @ x
            # Left-hand side vector
            lhs = P[i].T @ x
            constraints.append(cp.SOC(rhs, lhs))

        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        # Try Clarabel first (as in reference), then fall back to ECOS
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                # On failure, return unbounded-style output
                return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        # Check status
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        x_val = x.value
        if x_val is None or not np.all(np.isfinite(x_val)):
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        obj_val = float(c @ x_val)
        return {"objective_value": obj_val, "x": np.asarray(x_val, dtype=float)}