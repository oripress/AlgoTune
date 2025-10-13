from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Parse inputs
        G = np.asarray(problem["G"], dtype=float)
        sigma = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]
        if G.shape != (n, n):
            raise ValueError("G must be a square matrix")
        if not (sigma.shape == (n,) and P_min.shape == (n,) and P_max.shape == (n,)):
            raise ValueError("σ, P_min, P_max must be vectors of length n")

        # Build LP:
        # minimize    1^T P
        # subject to  A P >= b
        #             P_min <= P <= P_max
        #
        # From constraints:
        #   G_ii P_i >= S_min * (sigma_i + sum_k G_ik P_k - G_ii P_i)
        # =>
        #   (1+S_min) G_ii P_i - S_min * sum_k G_ik P_k >= S_min * sigma_i
        #
        # This simplifies to:
        #   A[i, i] = G_ii
        #   A[i, k != i] = -S_min * G_ik
        #   b[i] = S_min * sigma_i
        gdiag = np.diag(G).copy()
        # Start with A = -S_min * G, then fix diagonal by adding (1+S_min)*G_ii so that A[i,i] = G_ii
        A = -S_min * G.copy()
        A[np.arange(n), np.arange(n)] += (1.0 + S_min) * gdiag
        b = S_min * sigma

        # Convert A P >= b to -A P <= -b for linprog's standard form
        A_ub = -A
        b_ub = -b

        c = np.ones(n, dtype=float)
        bounds = list(zip(P_min.tolist(), P_max.tolist()))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success or res.x is None:
            raise ValueError(f"LP solver failed: {res.message}")

        P_opt = res.x
        obj = float(P_opt.sum())

        return {"P": P_opt.tolist(), "objective": obj}