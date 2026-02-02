from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Fortran order helps LAPACK (used by eigh) avoid an internal copy.
        A = np.zeros((n, n), dtype=np.float64, order="F")
        for i, neigh in enumerate(adj_list):
            if neigh:
                A[i, neigh] = 1.0

        try:
            w, v = np.linalg.eigh(A)
            # exp(A) = V diag(exp(w)) V^T = B B^T with B = V diag(exp(w/2))
            w *= 0.5
            np.exp(w, out=w)  # w := exp(w/2)
            v *= w  # scale columns in-place
            expA = v @ v.T
        except Exception:
            try:
                from scipy.linalg import expm  # type: ignore

                expA = expm(np.asarray(A))
            except Exception:
                return {"communicability": {}}

        rows = expA.tolist()
        inner = [dict(enumerate(r)) for r in rows]
        return {"communicability": dict(enumerate(inner))}