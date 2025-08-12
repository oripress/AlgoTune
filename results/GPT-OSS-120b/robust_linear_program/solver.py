import numpy as np
from typing import Any
import cvxpy as cp
class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solve the robust linear program with ellipsoidal uncertainty.
        The robust formulation reduces to a second‑order cone program:

            minimize    cᵀx
            subject to  q_iᵀx + ‖P_iᵀx‖₂ ≤ b_i   for all i

        Parameters
        ----------
        problem : dict
            "c": list[float] – objective coefficients (length n)
            "b": list[float] – RHS of constraints (length m)
            "P": list[list[list[float]]] – m matrices, each n×n (symmetric PSD)
            "q": list[list[float]] – m vectors, each length n

        Returns
        -------
        dict
            {"objective_value": float, "x": np.ndarray}
            If the problem is infeasible or unbounded, returns
            {"objective_value": inf, "x": np.full(n, np.nan)}.
        """
        # Convert inputs to NumPy arrays
        c = np.asarray(problem["c"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)   # shape (m, n, n)
        q = np.asarray(problem["q"], dtype=float)   # shape (m, n)

        m, n = len(b), len(c)

        # Primary solution method: CVXPY SOCP formulation
        try:
            import cvxpy as cp

            x = cp.Variable(n)

            # Pre‑compute transposes of P to avoid repeated work
            P_T = P.transpose(0, 2, 1)   # shape (m, n, n)

            soc_constraints = []
            for i in range(m):
                # CVXPY SOC: ‖P_iᵀ x‖₂ ≤ b_i - q_iᵀ x
                soc_constraints.append(cp.SOC(b[i] - q[i] @ x, P_T[i] @ x))

            prob = cp.Problem(cp.Minimize(c @ x), soc_constraints)

            # Try the fast ECOS solver first; fall back to CLARABEL if it fails
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status in ("optimal", "optimal_inaccurate"):
                x_opt = np.asarray(x.value, dtype=float)
                obj_val = float(c @ x_opt)
                return {"objective_value": obj_val, "x": x_opt}
            else:
                # Infeasible or unbounded according to CVXPY
                return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
        except Exception:
            # If CVXPY is unavailable or fails, indicate infeasibility
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}