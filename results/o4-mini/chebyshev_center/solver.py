from typing import Any, Dict
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, list]:
        """
        Compute the Chebyshev center of the polyhedron {x | a_i^T x <= b_i}.
        Solves the LP:
            maximize r
            s.t. a_i^T x + ||a_i|| * r <= b_i, for all i
                 r >= 0
        by converting to a standard form and using SciPy's HiGHS dual simplex.
        """
        # Load data
        a = np.array(problem["a"], dtype=float)
        b = np.array(problem["b"], dtype=float)
        if a.ndim != 2:
            raise ValueError("problem['a'] must be a 2D list")
        m, n = a.shape

        # Compute norms of each row a_i
        norms = np.linalg.norm(a, axis=1)

        # Objective: maximize r → minimize -r
        # Variable vector z = [x (n dims), r (1 dim)]
        c = np.zeros(n + 1, dtype=float)
        c[-1] = -1.0

        # Inequality constraints A_ub @ z <= b
        # For each i: a_i^T x + norms[i] * r <= b_i
        A_ub = np.hstack((a, norms.reshape(-1, 1)))

        # Bounds: x_j free, r >= 0
        bounds = [(None, None)] * n + [(0.0, None)]

        # Single HiGHS solve (disable presolve)
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"presolve": False},
        )
        if res.status != 0:
            raise RuntimeError(f"LP solver failed, status {res.status}: {res.message}")
        # Return the center x
        return {"solution": res.x[:n].tolist()}