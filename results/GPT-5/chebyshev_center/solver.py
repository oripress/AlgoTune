from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Compute the Chebyshev center (center of the largest inscribed ball) of a polyhedron
        P = {x | a_i^T x <= b_i, i = 1, ..., m}.

        This is solved via a linear program:
            maximize    r
            subject to  a_i^T x + r * ||a_i||_2 <= b_i  for all i
                        r >= 0

        :param problem: A dictionary with keys:
                        - "a": list of m lists (rows) representing the matrix A (constraints).
                        - "b": list of m numbers representing the vector b.
        :return: A dictionary with key "solution": the optimal center x_c as a list.
        """
        A = np.asarray(problem["a"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        if A.ndim != 2:
            raise ValueError("Input 'a' must be a 2D array-like.")
        if b.ndim != 1:
            raise ValueError("Input 'b' must be a 1D array-like.")
        m, n = A.shape
        if b.shape[0] != m:
            raise ValueError("Dimension mismatch: len(b) must equal number of rows in a.")

        # Row norms of A for the Chebyshev constraints
        s = np.linalg.norm(A, axis=1)

        # Variables z = [x (n vars); r (1 var)]
        # Objective: minimize -r (equivalent to maximize r)
        c = np.zeros(n + 1, dtype=float)
        c[-1] = -1.0

        # Inequalities: [A | s] @ [x; r] <= b
        A_ub = np.hstack([A, s.reshape(-1, 1)])
        b_ub = b

        # Bounds: x free, r >= 0
        bounds = [(None, None)] * n + [(0.0, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success or res.x is None:
            raise RuntimeError(f"Linear program failed: {getattr(res, 'message', 'unknown error')}")

        x_c = res.x[:n]
        if not np.all(np.isfinite(x_c)):
            raise RuntimeError("Non-finite solution encountered.")

        return {"solution": x_c.tolist()}