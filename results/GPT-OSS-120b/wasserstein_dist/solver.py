from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[float]]) -> float:
        """
        Compute the 1‑Wasserstein distance between two discrete 1‑D distributions.

        The distributions are given as probability (or mass) vectors `u` and `v`
        over the support {1, 2, ..., n}.  For the 1‑dimensional case the
        optimal transport cost reduces to the L1 distance between the
        cumulative distribution functions.

        Parameters
        ----------
        problem : dict
            Must contain keys ``"u"`` and ``"v"`` with equal‑length lists
            of non‑negative numbers.  The lists need not sum to 1; the
            algorithm works for any non‑negative mass vectors.

        Returns
        -------
        float
            The Wasserstein distance (a non‑negative float).
        """
        import numpy as np

        u = np.asarray(problem["u"], dtype=np.float64)
        v = np.asarray(problem["v"], dtype=np.float64)

        if u.shape != v.shape:
            raise ValueError("Distributions u and v must have the same length")

        # Difference and cumulative sum; exclude the last element (no cost beyond n)
        diff = u - v
        cum = np.cumsum(diff)[:-1]
        distance = np.abs(cum).sum()
        return float(distance)