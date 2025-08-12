import numpy as np

class Solver:
    def solve(self, problem):
        """
        Fast outer product via ufunc broadcasting with out:
        uses np.multiply for minimal overhead.
        """
        vec1, vec2 = problem
        a = np.asarray(vec1, dtype=np.float64)
        b = np.asarray(vec2, dtype=np.float64)
        n = a.shape[0]
        out = np.empty((n, n), dtype=np.float64)
        # broadcast multiply into preallocated array
        np.multiply(a[:, None], b, out=out)
        return out