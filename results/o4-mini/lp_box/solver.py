import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solve box-constrained LP: minimize c^T x s.t. A x <= b and 0 <= x <= 1.
        Uses trivial shortcuts and specialized routines for simple cases;
        otherwise uses SciPy's HiGHS solver.
        """
        c = problem["c"]
        A = problem["A"]
        b = problem["b"]
        n = len(c)
        # No constraints: only bounds
        if not A:
            # choose 1 for negative cost, else 0
            return {"solution": [1.0 if ci < 0 else 0.0 for ci in c]}
        m = len(A)
        # Zero objective is feasible: if all costs >=0 and 0 is feasible (b >= 0)
        if all(ci >= 0 for ci in c) and all(bj >= 0 for bj in b):
            return {"solution": [0.0] * n}
        # All costs non-positive and all-ones is feasible: choose all ones
        if all(ci <= 0 for ci in c):
            feasible_ones = True
            for j in range(m):
                row = A[j]
                if any(ai < 0 for ai in row) or sum(row) > b[j]:
                    feasible_ones = False
                    break
            if feasible_ones:
                return {"solution": [1.0] * n}
        # Single constraint: fractional knapsack when all coefficients >=0
        if m == 1:
            row = A[0]
            b0 = b[0]
            if b0 > 0 and all(ai >= 0 for ai in row):
                c_arr = np.asarray(c, dtype=float)
                a_arr = np.asarray(row, dtype=float)
                neg_mask = c_arr < 0
                if not np.any(neg_mask):
                    return {"solution": [0.0] * n}
                idx_neg = np.nonzero(neg_mask)[0]
                a_neg = a_arr[idx_neg]
                ratio = (c_arr[idx_neg] / a_neg)
                order = np.argsort(ratio)
                sorted_idx = idx_neg[order]
                sorted_a = a_arr[sorted_idx]
                cumsum = np.cumsum(sorted_a)
                # how many full items
                k = np.searchsorted(cumsum, b0, side='right')
                x = np.zeros(n, dtype=float)
                if k == 0:
                    x[sorted_idx[0]] = b0 / sorted_a[0]
                    return {"solution": x.tolist()}
                x[sorted_idx[:k]] = 1.0
                used = cumsum[k-1]
                if k < len(sorted_idx):
                    x[sorted_idx[k]] = (b0 - used) / sorted_a[k]
                return {"solution": x.tolist()}
        # General case: use HiGHS via SciPy
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0.0, 1.0), method="highs")
        if not res.success:
            # fallback to dual-simplex
            res = linprog(c, A_ub=A, b_ub=b, bounds=(0.0, 1.0), method="highs-ds")
            if not res.success:
                raise RuntimeError(f"LP solver failed: {res.message}")
        return {"solution": res.x.tolist()}