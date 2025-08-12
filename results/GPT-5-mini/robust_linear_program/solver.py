from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solve the robust LP:
            minimize    c^T x
            subject to  q_i^T x + ||P_i^T x||_2 <= b_i,  for i=1..m

        Inputs:
            problem: dict with keys "c", "b", "P", "q"
        Returns:
            dict with keys "objective_value" (float) and "x" (numpy.ndarray)
        """
        # Parse inputs
        try:
            c = np.asarray(problem["c"], dtype=float).ravel()
            b = np.asarray(problem["b"], dtype=float).ravel()
            P_list = problem.get("P", [])
            q_list = problem.get("q", [])
        except Exception:
            return {"objective_value": float("inf"), "x": np.array([np.nan])}

        n = int(c.size)
        m = int(b.size)

        # Handle trivial dimensions
        if n == 0:
            return {"objective_value": 0.0, "x": np.zeros(0, dtype=float)}
        if m == 0:
            # No constraints: unbounded unless objective is zero
            if np.allclose(c, 0.0, atol=1e-12):
                return {"objective_value": 0.0, "x": np.zeros(n, dtype=float)}
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}

        # Coerce P_i and q_i to correct shapes
        def make_P(pi):
            arr = np.asarray(pi, dtype=float)
            if arr.shape == (n, n):
                return arr
            if arr.size == n * n:
                return arr.ravel()[: n * n].reshape((n, n))
            if arr.ndim == 1 and arr.size == n:
                # treat as diagonal
                M = np.zeros((n, n), dtype=float)
                M[np.arange(n), np.arange(n)] = arr
                return M
            if arr.size == 1:
                return float(arr) * np.eye(n, dtype=float)
            # fallback: fill row-major
            flat = arr.ravel()
            M = np.zeros((n, n), dtype=float)
            ln = min(flat.size, n * n)
            if ln > 0:
                M.flat[:ln] = flat[:ln]
            return M

        def make_q(qi):
            arr = np.asarray(qi, dtype=float).ravel()
            if arr.size == n:
                return arr
            v = np.zeros(n, dtype=float)
            ln = min(arr.size, n)
            if ln > 0:
                v[:ln] = arr[:ln]
            return v

        P = [make_P(P_list[i]) if i < len(P_list) else np.zeros((n, n), dtype=float) for i in range(m)]
        q = [make_q(q_list[i]) if i < len(q_list) else np.zeros(n, dtype=float) for i in range(m)]

        # Solve using CVXPY (ECOS preferred for SOCP)
        try:
            import cvxpy as cp  # type: ignore

            x = cp.Variable(n)
            constraints = []
            for i in range(m):
                # constraint: q_i^T x + ||P_i^T x||_2 <= b_i
                rhs = float(b[i]) - q[i].T @ x
                constraints.append(cp.SOC(rhs, P[i].T @ x))

            prob = cp.Problem(cp.Minimize(c.T @ x), constraints)

            # Try ECOS first; fallback to default solver if necessary
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob.solve(verbose=False)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}

            x_val = np.asarray(x.value, dtype=float).ravel()
            if x_val.size != n or not np.all(np.isfinite(x_val)):
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}

            obj = float(np.dot(c, x_val))
            return {"objective_value": obj, "x": x_val}
        except Exception:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}