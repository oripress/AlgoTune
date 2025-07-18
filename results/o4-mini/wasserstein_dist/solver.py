import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the 1D Wasserstein distance between two discrete distributions u and v
        on support [1..n] via the sum of absolute differences of their CDFs.
        This matches scipy.stats.wasserstein_distance behavior by normalizing weights.
        """
        # Extract weight arrays
        u = np.asarray(problem.get("u", []), dtype=float)
        v = np.asarray(problem.get("v", []), dtype=float)
        # Normalize to total mass 1 (if nonzero), as scipy does
        su = u.sum()
        sv = v.sum()
        if su != 0.0:
            u /= su
        if sv != 0.0:
            v /= sv
        # CDFs and distance
        cdf_u = np.cumsum(u)
        cdf_v = np.cumsum(v)
        return float(np.abs(cdf_u - cdf_v).sum())