import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem, **kwargs):
        # Objective vector
        c = np.array(problem["c"], dtype=float)
        n = c.size

        # Bounds for x: 0 <= x_i <= 1
        bounds = [(0.0, 1.0)] * n

        # Get inequality constraints if present
        A_list = problem.get("A", [])
        b_list = problem.get("b", [])

        if A_list:
            A = np.array(A_list, dtype=float)
            b = np.array(b_list, dtype=float)
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        else:
            # No A x <= b constraints
            res = linprog(c, bounds=bounds, method="highs")

        # Numerical safety: clip to [0,1]
        x = np.clip(res.x, 0.0, 1.0)
        return {"solution": x.tolist()}