import cvxpy as cp
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves the minimum volume covering ellipsoid problem.
        
        Args:
            problem: Dictionary with "points" key containing array of points
            
        Returns:
            Dictionary with "objective_value" and "ellipsoid" (X, Y)
        """
        points = np.array(problem["points"])
        (n, d) = points.shape

        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable((d,))

        # Build constraints more efficiently using list comprehension
        constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]

        prob = cp.Problem(cp.Minimize(-cp.log_det(X)), constraints)

        try:
            # Use CLARABEL with tighter tolerance
            prob.solve(solver=cp.CLARABEL, verbose=False, tol_gap_abs=1e-7, tol_gap_rel=1e-7)

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }

            return {"objective_value": prob.value, "ellipsoid": {"X": X.value, "Y": Y.value}}

        except Exception:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }