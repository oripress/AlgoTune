from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a given minimum volume covering ellipsoid problem using CVXPY.
        """
        points = np.array(problem["points"])
        n, d = points.shape
        
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable((d,))
        
        # Create constraints - vectorized approach
        constraints = []
        for i in range(n):
            constraints.append(cp.SOC(1, X @ points[i] + Y))
        
        objective = cp.Minimize(-cp.log_det(X))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }
            
            return {
                "objective_value": prob.value,
                "ellipsoid": {"X": X.value, "Y": Y.value}
            }
        except Exception:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }