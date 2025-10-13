from __future__ import annotations

from typing import Any

import numpy as np
import ot

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the EMD problem using POT's linear programming solver, with
        optimized fast paths for 1xN, Nx1, and when one dimension equals 2.

        Returns
        -------
        dict
            {"transport_plan": np.ndarray of shape (n, m)}
        """
        # Convert to float64 numpy arrays for speed and consistency
        a = np.asarray(problem["source_weights"], dtype=np.float64).ravel()
        b = np.asarray(problem["target_weights"], dtype=np.float64).ravel()
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)

        n, m = a.size, b.size

        # Fast path: 1xN or Nx1 are trivial and unique
        if n == 1:
            G = np.zeros((1, m), dtype=np.float64)
            G[0, :] = b
            return {"transport_plan": G}
        if m == 1:
            G = np.zeros((n, 1), dtype=np.float64)
            G[:, 0] = a
            return {"transport_plan": G}

        # Fast path: when one dimension equals 2, the problem reduces to
        # sorting by cost differences and allocating greedily. This yields
        # the unique solution when differences are all distinct.
        tol = 1e-12

        if n == 2:
            # Minimize sum_j (M[0,j] - M[1,j]) * x_j + const
            # with 0 <= x_j <= b_j and sum_j x_j = a[0]
            d = M[0, :] - M[1, :]
            order = np.argsort(d, kind="mergesort")  # stable
            d_sorted = d[order]
            # If ties exist, fall back to POT to match its tie-breaking
            if np.any(np.isclose(d_sorted[1:], d_sorted[:-1], atol=tol, rtol=0.0)):
                M_cont = np.ascontiguousarray(M, dtype=np.float64)
                G = ot.lp.emd(a, b, M_cont, check_marginals=False)
                return {"transport_plan": G}
            rem = a[0]
            G = np.zeros((2, m), dtype=np.float64)
            last_idx = -1
            for j in order:
                if rem <= 0.0:
                    break
                bj = b[j]
                take = bj if rem >= bj else rem
                if take > 0.0:
                    G[0, j] = take
                    rem -= take
                    last_idx = j
            # Assign residuals to second row
            G[1, :] = b - G[0, :]
            # Correct any minute numerical residue due to floating ops
            if abs(rem) > tol and last_idx >= 0:
                G[0, last_idx] += rem
                G[1, last_idx] -= rem
            return {"transport_plan": G}

        if m == 2:
            # Minimize sum_i (M[i,0] - M[i,1]) * y_i + const
            # with 0 <= y_i <= a_i and sum_i y_i = b[0]
            d = M[:, 0] - M[:, 1]
            order = np.argsort(d, kind="mergesort")  # stable
            d_sorted = d[order]
            # If ties exist, fall back to POT to match its tie-breaking
            if np.any(np.isclose(d_sorted[1:], d_sorted[:-1], atol=tol, rtol=0.0)):
                M_cont = np.ascontiguousarray(M, dtype=np.float64)
                G = ot.lp.emd(a, b, M_cont, check_marginals=False)
                return {"transport_plan": G}
            rem = b[0]
            G = np.zeros((n, 2), dtype=np.float64)
            last_idx = -1
            for i in order:
                if rem <= 0.0:
                    break
                ai = a[i]
                take = ai if rem >= ai else rem
                if take > 0.0:
                    G[i, 0] = take
                    rem -= take
                    last_idx = i
            # Assign residuals to second column
            G[:, 1] = a - G[:, 0]
            # Correct any minute numerical residue due to floating ops
            if abs(rem) > tol and last_idx >= 0:
                G[last_idx, 0] += rem
                G[last_idx, 1] -= rem
            return {"transport_plan": G}

        # General case: use POT's network simplex (fast and reliable)
        M_cont = np.ascontiguousarray(M, dtype=np.float64)
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)
        return {"transport_plan": G}