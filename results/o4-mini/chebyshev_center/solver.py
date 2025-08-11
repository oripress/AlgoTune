import numpy as np
from typing import Any
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        """
        Compute the Chebyshev center of a polyhedron {x | a_i^T x <= b_i}.
        Solves the LP:
            maximize   r
            subject to a_i^T x + ||a_i|| * r <= b_i,  i=1..m
                       r >= 0
        :param problem: dict with keys "a" (m x n list) and "b" (m list)
        :return: dict with "solution": list of n floats (the center x)
        """
        a = np.asarray(problem["a"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        m, n = a.shape

        # Compute norms of each row a_i
        norms = np.linalg.norm(a, axis=1)

        # Objective: maximize r  <=> minimize -r
        c = np.zeros(n + 1, dtype=float)
        c[-1] = -1.0

        # Constraints: [a_i, norms[i]] * [x; r] <= b_i
        A_ub = np.hstack((a, norms.reshape(-1, 1)))
        b_ub = b

        # Bounds: x free, r >= 0
        bounds = [(None, None)] * n + [(0.0, None)]

        # Solve LP with HiGHS
        res = linprog(c,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      bounds=bounds,
                      method="highs-ds",
                      options={"presolve": False})

        if not res.success:
            raise RuntimeError(f"LP solver failed: {res.message}")

        x_opt = res.x[:n]
        return {"solution": x_opt.tolist()}