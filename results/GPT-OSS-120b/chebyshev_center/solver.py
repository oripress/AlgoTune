import numpy as np
from typing import Any, Dict, List
from scipy.optimize import linprog
from scipy import sparse

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Compute the Chebyshev center of a polyhedron defined by
        a_i^T x <= b_i for i = 1..m.

        The LP formulation:
            maximize    r
            subject to  a_i^T x + r * ||a_i||_2 <= b_i   for all i
                        r >= 0

        Returns a dictionary with key "solution" containing the optimal x (center).
        """
        # Convert inputs to numpy arrays
        a = np.asarray(problem["a"], dtype=float)   # shape (m, n)
        b = np.asarray(problem["b"], dtype=float)   # shape (m,)

        if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.shape[0]:
            raise ValueError("Invalid dimensions for 'a' or 'b'.")

        m, n = a.shape

        # Euclidean norms of each a_i (coefficients for r)
        norms = np.linalg.norm(a, axis=1)

        # Remove constraints with zero norm – they do not involve r
        nonzero = norms > 0
        a = a[nonzero]
        b = b[nonzero]
        norms = norms[nonzero]
        m = a.shape[0]

        # Objective: maximize r  => minimize -r
        c = np.zeros(n + 1)
        c[-1] = -1.0

        # Build inequality matrix [a | norms]  (A_ub @ [x; r] <= b)
        # Use dense matrix for compatibility with linprog
        A_ub = np.hstack([a, norms[:, None]])
        b_ub = b

        # Variable bounds: x free (no bounds), r >= 0
        bounds = [(None, None)] * n + [(0, None)]

        # Solve with HiGHS (fast interior‑point method)
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        # The solution vector is [x_center, r]; we return only x_center
        x_center = res.x[:n]
        return {"solution": x_center.tolist()}