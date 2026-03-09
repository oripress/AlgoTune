from typing import Any

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        v = np.asarray(problem.get("v")).ravel()
        n = v.size
        k = problem.get("k")

        # Match NumPy slice semantics of reference: indx[-k:]
        if k == 0:
            return {"solution": v}
        if k > 0:
            m = k if k < n else n
        else:
            m = n + k
            if m < 0:
                m = 0

        if m <= 0:
            return {"solution": np.zeros_like(v)}
        if m >= n:
            return {"solution": v}

        q = n - m
        a = np.abs(v)
        t = np.partition(a, q)[q]

        # Write fewer elements depending on sparsity level.
        if m <= q:
            keep = a > t
            need = m - int(np.count_nonzero(keep))
            if need > 0:
                eq_idx = np.nonzero(a == t)[0]
                keep[eq_idx[-need:]] = True
        else:
            keep = a >= t
            drop = int(np.count_nonzero(keep)) - m
            if drop > 0:
                eq_idx = np.nonzero(a == t)[0]
                keep[eq_idx[:drop]] = False

        return {"solution": v * keep}