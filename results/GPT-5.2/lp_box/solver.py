from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    # SciPy's HiGHS interface is typically far faster than CVXPY for LPs.
    from scipy.optimize import linprog  # type: ignore
except Exception:  # pragma: no cover
    linprog = None  # type: ignore

class Solver:
    def __init__(self) -> None:
        # Keep init lightweight (init time not counted anyway).
        pass

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, List[float]]:
        """
        LP Box:
            minimize    c^T x
            subject to  A x <= b
                        0 <= x <= 1
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Shape hardening for corner cases.
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if b.ndim == 0:
            b = b.reshape(1)

        n = int(c.shape[0])
        if n == 0:
            return {"solution": []}

        # No linear constraints: solution is just box optimum.
        if A.size == 0:
            x = (c < 0.0).astype(np.float64, copy=False)
            return {"solution": x.tolist()}

        # Fast-path: box-optimum (by sign of c) if already feasible for Ax<=b.
        # For min c^T x with 0<=x<=1, a minimizer is x_i=1 when c_i<0 else 0.
        x0 = (c < 0.0).astype(np.float64, copy=False)

        tol = 1e-9
        if not x0.any():
            # A@0 = 0
            if np.all(b >= -tol):
                return {"solution": x0.tolist()}
        elif x0.all():
            # A@1 = row sums
            if np.all(np.sum(A, axis=1) <= b + tol):
                return {"solution": x0.tolist()}
        else:
            if np.all(A @ x0 <= b + tol):
                return {"solution": x0.tolist()}

        # Primary solver: HiGHS via SciPy.
        if linprog is not None:
            highs_opts = {
                "presolve": True,
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "ipm_optimality_tolerance": 1e-9,
                "output_flag": False,
            }
            try:
                res = linprog(
                    c,
                    A_ub=A,
                    b_ub=b,
                    bounds=(0.0, 1.0),
                    method="highs",
                    options=highs_opts,
                )
            except Exception:
                # Defensive: option naming may differ across SciPy/HiGHS builds.
                res = linprog(
                    c,
                    A_ub=A,
                    b_ub=b,
                    bounds=(0.0, 1.0),
                    method="highs",
                )

            if res is not None and res.status == 0 and res.x is not None:
                x = res.x.astype(np.float64, copy=False)
                if not np.all(np.isfinite(x)):
                    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                return {"solution": x.tolist()}

        # Fallback: CVXPY+Clarabel for robustness.
        import cvxpy as cp  # type: ignore

        x_var = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c @ x_var), [A @ x_var <= b, 0 <= x_var, x_var <= 1])
        prob.solve(solver="CLARABEL")
        x_val = np.asarray(x_var.value, dtype=np.float64)
        if not np.all(np.isfinite(x_val)):
            x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
        return {"solution": x_val.tolist()}